#include "ggml.h"
#include "whisper.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <algorithm>
#include <queue>

constexpr int N_THREAD = 8;

std::vector<struct whisper_context *> g_contexts(4, nullptr);

std::mutex g_mutex;
std::thread g_worker;

std::atomic<bool> g_running(false);

std::string g_status        = "";
std::string g_status_forced = "";
std::string g_transcribed   = "";
std::string last_transcribed = "";

struct Segment {
    std::string text;
    int64_t start_time;  // Start time of the segment
    int64_t end_time;    // End time of the segment
};

struct AudioChunk {
    std::vector<float> pcmf32;
    bool is_final;
};

std::vector<AudioChunk> g_audio_chunks;

std::string normalize_string(const std::string &str) {
    std::string result;
    for (char c : str) {
        if (!std::ispunct(c)) {  // Remove punctuation
            result += std::tolower(c);  // Convert to lowercase
        }
    }
    return result;
}

int levenshtein_distance(const std::string &s1, const std::string &s2) {
    const size_t len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<int>> d(len1 + 1, std::vector<int>(len2 + 1));

    for (size_t i = 0; i <= len1; i++) d[i][0] = i;
    for (size_t j = 0; j <= len2; j++) d[0][j] = j;

    for (size_t i = 1; i <= len1; i++) {
        for (size_t j = 1; j <= len2; j++) {
            d[i][j] = std::min({
                d[i - 1][j] + 1,  // Deletion
                d[i][j - 1] + 1,  // Insertion
                d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1) // Substitution
            });
        }
    }

    return d[len1][len2];
}

bool words_match(const std::string &s1, const std::string &s2, bool levenshtein_fallback) {
    if (s1 == s2) return true;

    std::string norm_s1 = normalize_string(s1);
    std::string norm_s2 = normalize_string(s2);

    if (norm_s1 == norm_s2) return true;

    if (levenshtein_fallback) {
        return levenshtein_distance(norm_s1, norm_s2) <= 1;  // Allow 1 small change
    }

    return false;
}

std::vector<std::string> split_words(const std::string &text) {
    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

std::string join_words(const std::vector<std::string> &words, size_t start = 0) {
    std::ostringstream result;
    for (size_t i = start; i < words.size(); i++) {
        if (i > start) result << " ";
        result << words[i];
    }
    return result.str();
}

std::string deduplicate_transcription(const std::string &new_text, bool is_final) {
    std::vector<std::string> new_words = split_words(new_text);

    // Remove last word from transcription to avoid cutoff words
    if (!is_final && !new_words.empty()) {
        new_words.pop_back();
    }

    if (last_transcribed.empty()) {
        last_transcribed = join_words(new_words);
        return last_transcribed;
    }

    std::vector<std::string> old_words = split_words(last_transcribed);

    // Search BACKWARD for the first matching word
    size_t match_index = old_words.size();
    std::string first_new_word = new_words.empty() ? "" : new_words[0];

    for (size_t i = old_words.size(); i-- > 0;) {  // Iterate backward
        if (words_match(old_words[i], first_new_word, false)) { // Exact match only for the first word
            match_index = i;
            break;
        }
    }

    // Move FORWARD and remove overlapping words
    size_t trim_index = 0;
    while (trim_index < new_words.size() && match_index < old_words.size()) {
        if (words_match(old_words[match_index], new_words[trim_index], true)) { // Exact match or levenshtein for overlapping words after first match
            trim_index++;  // Remove this word
            match_index++;
        } else {
            break;  // Stop at first non-match
        }
    }

    // Trim overlapping words
    std::vector<std::string> deduped_words(new_words.begin() + trim_index, new_words.end());

    // Store cleaned transcription & return result
    if (is_final) {
        last_transcribed.clear(); // Clear previous transcription
        g_audio_chunks.clear(); // Clear the 1-second buffer AFTER processing the final frame
    } else {
        last_transcribed = join_words(new_words);
    }
    return join_words(deduped_words);
}

void stream_main(size_t index) {
    struct whisper_full_params wparams = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);

    wparams.n_threads        = std::min(N_THREAD, (int) std::thread::hardware_concurrency());
    wparams.offset_ms        = 0;
    wparams.translate        = false;
    wparams.no_context       = true;
    wparams.single_segment   = true;
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.print_special    = false;

    wparams.max_tokens       = 32;
    wparams.audio_ctx        = 768; // partial encoder context for better performance
    wparams.temperature_inc  = -1.0f; // disable temperature fallback
    wparams.language         = "en";

    printf("stream: using %d threads\n", wparams.n_threads);

    std::vector<float> pcmf32;

    // whisper context
    auto & ctx = g_contexts[index];

    while (g_running) {
        {
            std::unique_lock<std::mutex> lock(g_mutex);
    
            if (g_audio_chunks.empty()) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
    
            // Get next chunk
            AudioChunk chunk = std::move(g_audio_chunks.front());
            g_audio_chunks.erase(g_audio_chunks.begin());
    
            lock.unlock();
    
            // Sliding Buffer Logic
            std::vector<float> new_pcmf32;
            const int64_t overlap_samples = WHISPER_SAMPLE_RATE;  // 1s prepended overlap
    
            if (!pcmf32.empty() && pcmf32.size() > overlap_samples) {
                new_pcmf32.insert(new_pcmf32.end(), pcmf32.end() - overlap_samples, pcmf32.end());
            }
    
            new_pcmf32.insert(new_pcmf32.end(), chunk.pcmf32.begin(), chunk.pcmf32.end());
    
            pcmf32 = std::move(new_pcmf32);
    
            int ret = whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size());
            if (ret != 0) {
                printf("whisper_full() failed: %d\n", ret);
                break;
            }
    
            // Transcription processing
            std::string text_heard;
            {
                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = n_segments - 1; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);
                    text_heard += text;
                }
            }
    
            text_heard = deduplicate_transcription(text_heard, chunk.is_final);
            printf("whisper: %s\n", text_heard.c_str());

            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_transcribed = text_heard;
            }
    
            // Clear buffer if final frame
            if (chunk.is_final) {
                pcmf32.clear();
            }
        }
    }

    if (index < g_contexts.size()) {
        whisper_free(g_contexts[index]);
        g_contexts[index] = nullptr;
    }
}

EMSCRIPTEN_BINDINGS(stream) {
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model) {
        for (size_t i = 0; i < g_contexts.size(); ++i) {
            if (g_contexts[i] == nullptr) {
                g_contexts[i] = whisper_init_from_file_with_params(path_model.c_str(), whisper_context_default_params());
                if (g_contexts[i] != nullptr) {
                    g_running = true;
                    if (g_worker.joinable()) {
                        g_worker.join();
                    }
                    g_worker = std::thread([i]() {
                        stream_main(i);
                    });

                    return i + 1;
                } else {
                    return (size_t) 0;
                }
            }
        }

        return (size_t) 0;
    }));

    emscripten::function("free", emscripten::optional_override([](size_t index) {
        if (g_running) {
            g_running = false;
        }
    }));

    emscripten::function("set_audio", emscripten::optional_override([](size_t index, const emscripten::val & audio, bool is_final) {
        --index;

        if (index >= g_contexts.size()) {
            return -1;
        }

        if (g_contexts[index] == nullptr) {
            return -2;
        }

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            const int n = audio["length"].as<int>();
    
            emscripten::val heap = emscripten::val::module_property("HEAPU8");
            emscripten::val memory = heap["buffer"];
    
            std::vector<float> pcmf32(n);
            emscripten::val memoryView = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
            memoryView.call<void>("set", audio);
    
            // Push audio + is_final flag as a single struct
            g_audio_chunks.push_back({std::move(pcmf32), is_final});
        }

        return 0;
    }));

    emscripten::function("get_transcribed", emscripten::optional_override([]() {
        std::string transcribed;

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            transcribed = std::move(g_transcribed);
        }

        return transcribed;
    }));
}
