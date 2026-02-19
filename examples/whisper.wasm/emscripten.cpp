#include "whisper.h"

#include <emscripten.h>
#include <emscripten/bind.h>

#include <vector>
#include <thread>
#include <string>
#include <sstream>

std::thread g_worker;

// Escape for JSON string values
static std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else out += c;
    }
    return out;
}

std::vector<struct whisper_context *> g_contexts(4, nullptr);

static inline int mpow2(int n) {
    int p = 1;
    while (p <= n) p *= 2;
    return p/2;
}

EMSCRIPTEN_BINDINGS(whisper) {
    emscripten::function("init", emscripten::optional_override([](const std::string & path_model) {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        for (size_t i = 0; i < g_contexts.size(); ++i) {
            if (g_contexts[i] == nullptr) {
                g_contexts[i] = whisper_init_from_file_with_params(path_model.c_str(), whisper_context_default_params());
                if (g_contexts[i] != nullptr) {
                    return i + 1;
                } else {
                    return (size_t) 0;
                }
            }
        }

        return (size_t) 0;
    }));

    emscripten::function("free", emscripten::optional_override([](size_t index) {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        --index;

        if (index < g_contexts.size()) {
            whisper_free(g_contexts[index]);
            g_contexts[index] = nullptr;
        }
    }));

    emscripten::function("full_default", emscripten::optional_override([](size_t index, const emscripten::val & audio, const std::string & lang, int nthreads, bool translate) -> std::string {
        if (g_worker.joinable()) {
            g_worker.join();
        }

        --index;

        if (index >= g_contexts.size()) {
            return "{\"status\":-1,\"segments\":[]}";
        }

        if (g_contexts[index] == nullptr) {
            return "{\"status\":-2,\"segments\":[]}";
        }

        struct whisper_full_params params = whisper_full_default_params(whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY);

        params.print_realtime   = true;
        params.print_progress   = false;
        params.print_timestamps  = true;
        params.print_special    = false;
        params.translate        = translate;
        params.language         = whisper_is_multilingual(g_contexts[index]) ? lang.c_str() : "en";
        params.n_threads        = std::min(nthreads, std::min(16, mpow2(std::thread::hardware_concurrency())));
        params.offset_ms        = 0;
        // Word-level segments: one segment per word with start/end timestamps
        params.token_timestamps = true;
        params.split_on_word    = true;
        params.max_len          = 1;

        std::vector<float> pcmf32;
        const int n = audio["length"].as<int>();

        emscripten::val heap = emscripten::val::module_property("HEAPU8");
        emscripten::val memory = heap["buffer"];

        pcmf32.resize(n);

        emscripten::val memoryView = audio["constructor"].new_(memory, reinterpret_cast<uintptr_t>(pcmf32.data()), n);
        memoryView.call<void>("set", audio);

        // print system information
        {
            printf("system_info: n_threads = %d / %d | %s\n",
                    params.n_threads, std::thread::hardware_concurrency(), whisper_print_system_info());

            printf("%s: processing %d samples, %.1f sec, %d threads, %d processors, lang = %s, task = %s ...\n",
                    __func__, int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE,
                    params.n_threads, 1,
                    params.language,
                    params.translate ? "translate" : "transcribe");

            printf("\n");
        }

#ifdef __EMSCRIPTEN__
        // Run on main thread: g_worker.join() hangs (pthread worker never signals completion / deadlock).
        // Single-threaded so we don't spawn a pthread at all.
        params.n_threads = 1;
        printf("full_default: [emscripten] running whisper_full on main thread (no pthread)\n");
        fflush(stdout);
        whisper_reset_timings(g_contexts[index]);
        whisper_full(g_contexts[index], params, pcmf32.data(), pcmf32.size());
        printf("full_default: whisper_full returned, printing timings\n");
        fflush(stdout);
        whisper_print_timings(g_contexts[index]);
        printf("full_default: collecting segments\n");
        fflush(stdout);
#else
        printf("full_default: starting worker thread\n");
        fflush(stdout);
        g_worker = std::thread([index, params, pcmf32 = std::move(pcmf32)]() {
            printf("full_default worker: thread started, calling whisper_full\n");
            fflush(stdout);
            whisper_reset_timings(g_contexts[index]);
            whisper_full(g_contexts[index], params, pcmf32.data(), pcmf32.size());
            printf("full_default worker: whisper_full returned, printing timings\n");
            fflush(stdout);
            whisper_print_timings(g_contexts[index]);
            printf("full_default worker: thread exiting\n");
            fflush(stdout);
        });
        printf("full_default: calling g_worker.join() ...\n");
        fflush(stdout);
        g_worker.join();
        printf("full_default: join() returned, collecting segments\n");
        fflush(stdout);
#endif

        // t0/t1 from whisper API are in centiseconds (100 = 1 sec); convert to seconds for JSON
        auto * ctx = g_contexts[index];
        const int n_segments = whisper_full_n_segments(ctx);
        printf("full_default: n_segments = %d\n", n_segments);
        fflush(stdout);
        std::ostringstream json;
        json << "{\"status\":0,\"segments\":[";
        for (int i = 0; i < n_segments; ++i) {
            const char * text = whisper_full_get_segment_text(ctx, i);
            const int64_t t0  = whisper_full_get_segment_t0(ctx, i);
            const int64_t t1  = whisper_full_get_segment_t1(ctx, i);
            if (i > 0) json << ",";
            json << "{\"text\":\"" << json_escape(text ? text : "")
                 << "\",\"t0\":" << (t0 / 100.0)
                 << ",\"t1\":" << (t1 / 100.0) << "}";
        }
        json << "]}";
        printf("full_default: built JSON (%zu bytes), returning\n", json.str().size());
        fflush(stdout);
        return json.str();
    }));
}
