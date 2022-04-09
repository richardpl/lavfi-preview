#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imnodes.h"

#include <stdio.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers
#include <unistd.h>

#include <AL/alc.h>
#include <AL/al.h>
#include <AL/alext.h>

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/avstring.h>
#include <libavutil/bprint.h>
#include <libavutil/dict.h>
#include <libavutil/opt.h>
#include <libavutil/parseutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/time.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavformat/avformat.h>
#include "ringbuffer/ringbuffer.c"
}

#define AL_BUFFERS 16

typedef struct FrameInfo {
    int width, height;
    int nb_samples;
    int format;
    int key_frame;
    enum AVPictureType pict_type;
    AVRational sample_aspect_ratio;
    int64_t pts;
    int64_t pkt_dts;
    AVRational time_base;
    int coded_picture_number;
    int display_picture_number;
    int interlaced_frame;
    int top_field_first;
    int sample_rate;
    enum AVColorRange color_range;
    enum AVColorPrimaries color_primaries;
    enum AVColorTransferCharacteristic color_trc;
    enum AVColorSpace colorspace;
    enum AVChromaLocation chroma_location;
    int64_t pkt_pos;
    int64_t pkt_duration;
    int pkt_size;
    size_t crop_top;
    size_t crop_bottom;
    size_t crop_left;
    size_t crop_right;
} FrameInfo;

typedef struct Edge2Pad {
    unsigned node;
    bool removed;
    bool is_output;
    unsigned pad_index;
    enum AVMediaType type;
} Edge2Pad;

typedef struct OptStorage {
    union {
        int i32;
        float flt;
        int64_t i64;
        uint64_t u64;
        double dbl;
        AVRational q;
        char *str;
    } u;
} OptStorage;

typedef struct BufferSink {
    unsigned id;
    char *label;
    AVFilterContext *ctx;
    AVRational time_base;
    AVRational frame_rate;
    ring_buffer_t consume_frames;
    ring_buffer_t render_frames;
    ring_buffer_t purge_frames;
    unsigned render_ring_size;
    double speed;
    bool ready;
    bool fullscreen;
    bool muted;
    bool show_osd;
    bool have_window_pos;
    ImVec2 window_pos;
    GLuint texture;
    ALuint source;
    ALenum format;
    float gain;
    float position[3];
    ALuint buffers[AL_BUFFERS];
    std::vector<ALuint> processed_bufids;
    std::vector<ALuint> unprocessed_bufids;

    GLint downscale_interpolator;
    GLint upscale_interpolator;

    int64_t frame_number;
    int64_t delta;
    int64_t qpts;
    int64_t pts;
    int64_t pos;
    int frame_nb_samples;

    float *samples;
    unsigned nb_samples;
    unsigned sample_index;
} BufferSink;

typedef struct FilterNode {
    unsigned id;
    bool set_pos;
    ImVec2 pos;
    int edge;
    bool colapsed;
    const AVFilter *filter;
    char *filter_name;
    char *filter_label;
    char *ctx_options;
    char *filter_options;
    AVFilterContext *probe;
    AVFilterContext *ctx;

    std::vector<int> inpad_edges;
    std::vector<int> outpad_edges;

    std::vector<OptStorage> opt_storage;
} FilterNode;

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

bool full_screen = false;
bool restart_display = false;
int filter_graph_nb_threads = 0;
int filter_graph_auto_convert_flags = 0;
unsigned focus_buffersink_window = -1;
unsigned focus_abuffersink_window = -1;
bool show_abuffersink_window = true;
bool show_buffersink_window = true;
bool show_dumpgraph_window = false;
bool show_commands_window = false;
bool show_filtergraph_editor_window = true;
bool show_mini_map = true;
int mini_map_location = ImNodesMiniMapLocation_BottomRight;

char *import_script_file_name = NULL;

bool need_filters_reinit = true;
bool framestep = false;
bool paused = true;
bool show_info = false;
bool show_help = false;
bool show_console = false;
bool show_log_window = false;

int log_level = AV_LOG_INFO;
ImGuiTextBuffer log_buffer;
ImVector<int> log_lines_offsets;
std::mutex log_mutex;

GLint global_upscale_interpolation = GL_NEAREST;
GLint global_downscale_interpolation = GL_NEAREST;

int output_sample_rate = 44100;
int display_w;
int display_h;
int width = 1280;
int height = 720;
bool filter_graph_is_valid = false;
AVFilterGraph *filter_graph = NULL;
AVFilterGraph *probe_graph = NULL;
char *graphdump_text = NULL;
float audio_sample_range[2] = { 1.f, 1.f };
float audio_window_size[2] = { 0, 100 };

int editor_edge = 0;
ImNodesEditorContext *node_editor_context;

std::mutex filtergraph_mutex;

std::vector<ALuint> play_sources;
std::thread play_sound_thread;

std::vector<BufferSink> abuffer_sinks;
std::vector<BufferSink> buffer_sinks;
std::vector<std::condition_variable> acv;
std::vector<std::condition_variable> cv;
std::vector<std::mutex> amutexes;
std::vector<std::mutex> mutexes;
std::vector<std::thread> audio_sink_threads;
std::vector<std::thread> video_sink_threads;
std::vector<FilterNode> filter_nodes;
std::vector<std::pair<int, int>> filter_links;
std::vector<Edge2Pad> edge2pad;

static const enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_RGBA, AV_PIX_FMT_NONE };
static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_NONE };
static const int sample_rates[] = { output_sample_rate, 0 };

ALCdevice *al_dev = NULL;
ALCcontext *al_ctx = NULL;
float listener_direction[6] = { 0, 0, -1, 0, 1, 0 };
float listener_position[3] = { 0, 0, 0 };

FrameInfo frame_info;

static void clear_ring_buffer(ring_buffer_t *ring_buffer)
{
    while (ring_buffer_num_items(ring_buffer) > 0) {
        AVFrame *frame;

        ring_buffer_dequeue(ring_buffer, &frame);
        av_frame_free(&frame);
    }
}

static void sound_thread(ALsizei nb_sources, std::vector<ALuint> *sources)
{
    bool state = paused && !framestep;

    if (state)
        alSourceStopv(nb_sources, sources->data());

    while (!need_filters_reinit && filter_graph_is_valid) {
        bool new_state = paused && !framestep;

        if (sources->size() == 0)
            break;

        if (state != new_state) {
            state = new_state;
            if (state == true)
                alSourcePausev(nb_sources, sources->data());
            else
                alSourcePlayv(nb_sources, sources->data());
        }
        av_usleep(10000);
    }
}

static void worker_thread(BufferSink *sink, std::mutex *mutex, std::condition_variable *cv)
{
    int ret;

    while (sink->ctx) {
        std::unique_lock lk(*mutex);
        cv->wait(lk, [sink]{ return sink->ready == true; });
        if (need_filters_reinit)
            break;
        if (sink->ready == false)
            continue;
        sink->ready = false;

        if (ring_buffer_num_items(&sink->consume_frames) < 1) {
            AVFrame *filter_frame;
            int64_t start, end;

            filter_frame = av_frame_alloc();
            if (!filter_frame)
                break;
            start = av_gettime_relative();
            filtergraph_mutex.lock();
            ret = av_buffersink_get_frame_flags(sink->ctx, filter_frame, 0);
            filtergraph_mutex.unlock();
            end = av_gettime_relative();
            if (end > start && filter_frame)
                sink->speed = 1000000. * (std::max(filter_frame->nb_samples, 1)) * av_q2d(av_inv_q(sink->frame_rate)) / (end - start);
            if (ret < 0 && ret != AVERROR(EAGAIN)) {
                av_frame_free(&filter_frame);
                break;
            }

            ring_buffer_enqueue(&sink->consume_frames, filter_frame);
        }

        if (paused)
            av_usleep(100000);
    }

    clear_ring_buffer(&sink->consume_frames);
    clear_ring_buffer(&sink->render_frames);
    clear_ring_buffer(&sink->purge_frames);
}

static void kill_audio_sink_threads()
{
    for (unsigned i = 0; i < audio_sink_threads.size(); i++) {
        BufferSink *sink = &abuffer_sinks[i];

        if (audio_sink_threads[i].joinable()) {
            { std::lock_guard lk(amutexes[i]); sink->ready = true; }
            acv[i].notify_one();
            audio_sink_threads[i].join();
        }

        av_freep(&sink->label);
        av_freep(&sink->samples);
        alDeleteSources(1, &sink->source);
        alDeleteBuffers(AL_BUFFERS, sink->buffers);
    }
}

static void kill_video_sink_threads()
{
    for (unsigned i = 0; i < video_sink_threads.size(); i++) {
        BufferSink *sink = &buffer_sinks[i];

        if (video_sink_threads[i].joinable()) {
            { std::lock_guard lk(mutexes[i]); sink->ready = true; }
            cv[i].notify_one();
            video_sink_threads[i].join();
        }

        av_freep(&sink->label);
        glDeleteTextures(1, &sink->texture);
    }
}

static int get_nb_filter_threads(const AVFilter *filter)
{
    if (filter->flags & AVFILTER_FLAG_SLICE_THREADS)
        return 0;
    return 1;
}

static int filters_setup()
{
    const AVFilter *new_filter;
    int ret;

    if (need_filters_reinit == false)
        return 0;

    if (play_sound_thread.joinable())
        play_sound_thread.join();
    play_sources.clear();

    kill_audio_sink_threads();
    kill_video_sink_threads();

    audio_sink_threads.clear();
    video_sink_threads.clear();

    need_filters_reinit = false;
    filter_graph_is_valid = false;

    if (filter_nodes.size() == 0)
        return 0;

    buffer_sinks.clear();
    abuffer_sinks.clear();
    cv.clear();
    acv.clear();
    mutexes.clear();
    amutexes.clear();

    av_freep(&graphdump_text);

    avfilter_graph_free(&filter_graph);
    filter_graph = avfilter_graph_alloc();
    if (!filter_graph) {
        av_log(NULL, AV_LOG_ERROR, "Cannot allocate filter graph.\n");
        ret = AVERROR(ENOMEM);
        goto error;
    }

    filter_graph->nb_threads = filter_graph_nb_threads;
    avfilter_graph_set_auto_convert(filter_graph, filter_graph_auto_convert_flags);

    for (unsigned i = 0; i < filter_nodes.size(); i++) {
        AVFilterContext *filter_ctx;

        new_filter = filter_nodes[i].filter;
        if (!new_filter) {
            av_log(NULL, AV_LOG_ERROR, "Cannot [%d] get filter by name: %s.\n", i, filter_nodes[i].filter_name);
            ret = AVERROR(ENOSYS);
            goto error;
        }

        filter_ctx = avfilter_graph_alloc_filter(filter_graph, new_filter, filter_nodes[i].filter_label);
        if (!filter_ctx) {
            av_log(NULL, AV_LOG_ERROR, "Cannot allocate filter context.\n");
            ret = AVERROR(ENOMEM);
            goto error;
        }

        av_opt_set_defaults(filter_ctx);
        filter_ctx->nb_threads = get_nb_filter_threads(filter_ctx->filter);

        if (!strcmp(filter_ctx->filter->name, "buffersink")) {
            BufferSink new_sink;

            new_sink.ctx = filter_ctx;
            new_sink.ready = false;
            new_sink.have_window_pos = false;
            new_sink.fullscreen = false;
            new_sink.muted = false;
            new_sink.show_osd = true;
            new_sink.frame_number = 0;
            new_sink.upscale_interpolator = global_upscale_interpolation;
            new_sink.downscale_interpolator = global_downscale_interpolation;
            ret = av_opt_set_int_list(filter_ctx, "pix_fmts", pix_fmts,
                                      AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Cannot set buffersink output pixel format.\n");
                goto error;
            }

            buffer_sinks.push_back(new_sink);
        } else if (!strcmp(filter_ctx->filter->name, "abuffersink")) {
            BufferSink new_sink;

            new_sink.ctx = filter_ctx;
            new_sink.ready = false;
            new_sink.have_window_pos = false;
            new_sink.fullscreen = false;
            new_sink.muted = false;
            new_sink.show_osd = true;
            new_sink.upscale_interpolator = 0;
            new_sink.downscale_interpolator = 0;
            new_sink.frame_number = 0;
            ret = av_opt_set_int_list(filter_ctx, "sample_fmts", sample_fmts,
                                      AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Cannot set abuffersink output sample formats.\n");
                goto error;
            }

            ret = av_opt_set_int_list(filter_ctx, "sample_rates", sample_rates,
                                      0, AV_OPT_SEARCH_CHILDREN);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Cannot set abuffersink output sample rates.\n");
                goto error;
            }

            abuffer_sinks.push_back(new_sink);
        }

        filter_nodes[i].ctx = filter_ctx;

        av_freep(&filter_nodes[i].ctx_options);
        if (filter_nodes[i].probe) {
            ret = av_opt_serialize(filter_nodes[i].probe, 0, AV_OPT_SERIALIZE_SKIP_DEFAULTS,
                                   &filter_nodes[i].ctx_options, '=', ':');
            if (ret < 0)
                av_log(NULL, AV_LOG_WARNING, "Cannot serialize filter ctx options.\n");
        }

        av_freep(&filter_nodes[i].filter_options);
        if (filter_nodes[i].probe) {
            ret = av_opt_serialize(filter_nodes[i].probe->priv, AV_OPT_FLAG_FILTERING_PARAM, AV_OPT_SERIALIZE_SKIP_DEFAULTS,
                                   &filter_nodes[i].filter_options, '=', ':');
            if (ret < 0)
                av_log(NULL, AV_LOG_WARNING, "Cannot serialize filter private options.\n");
        }

        ret = av_opt_set_from_string(filter_ctx, filter_nodes[i].ctx_options, NULL, "=", ":");
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Error setting filter ctx options.\n");
            goto error;
        }

        ret = av_opt_set_from_string(filter_ctx->priv, filter_nodes[i].filter_options, NULL, "=", ":");
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Error setting filter private options.\n");
            goto error;
        }

        ret = avfilter_init_str(filter_ctx, NULL);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Cannot init str for filter.\n");
            goto error;
        }
    }

    for (unsigned i = 0; i < filter_links.size(); i++) {
        const std::pair<int, int> p = filter_links[i];

        if ((unsigned)p.first >= edge2pad.size() ||
            (unsigned)p.second >= edge2pad.size()) {
            av_log(NULL, AV_LOG_ERROR, "Cannot link filters: edges out of range (%d, %d) >= (%ld).\n", p.first, p.second, edge2pad.size());
            ret = AVERROR(EINVAL);
            goto error;
        }

        unsigned x = edge2pad[p.first].node;
        unsigned y = edge2pad[p.second].node;
        unsigned x_pad = edge2pad[p.first].pad_index;
        unsigned y_pad = edge2pad[p.second].pad_index;
        bool x_out = edge2pad[p.first].is_output;
        bool y_out = edge2pad[p.second].is_output;

        if (x >= filter_nodes.size() || y >= filter_nodes.size()) {
            av_log(NULL, AV_LOG_ERROR, "Cannot link filters: index (%d, %d) out of range (%ld)\n",
                   x, y, filter_nodes.size());
            ret = AVERROR(EINVAL);
            goto error;
        }

        if (y_out == true) {
            std::swap(x, y);
            std::swap(x_pad, y_pad);
            std::swap(x_out, y_out);
        }

        if ((ret = avfilter_link(filter_nodes[x].ctx, x_pad, filter_nodes[y].ctx, y_pad)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "Cannot link filters: %s(%d|%d) <-> %s(%d|%d)\n",
                   filter_nodes[x].filter_label, x_pad, x_out, filter_nodes[y].filter_label, y_pad, y_out);
            goto error;
        }
    }

    if ((ret = avfilter_graph_config(filter_graph, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot configure graph.\n");
        goto error;
    }

    filter_graph_is_valid = true;
    framestep = false;
    paused = true;

    graphdump_text = avfilter_graph_dump(filter_graph, NULL);

    show_abuffersink_window = true;
    show_buffersink_window = true;

error:

    if (ret < 0)
        return ret;

    std::vector<std::condition_variable> cv_list(buffer_sinks.size());
    cv.swap(cv_list);

    std::vector<std::mutex> mutex_list(buffer_sinks.size());
    mutexes.swap(mutex_list);

    std::vector<std::thread> thread_list(buffer_sinks.size());
    video_sink_threads.swap(thread_list);

    for (unsigned i = 0; i < buffer_sinks.size(); i++) {
        BufferSink *sink = &buffer_sinks[i];

        sink->id = i;
        sink->label = av_asprintf("Video FilterGraph Output %d", i);
        sink->time_base = av_buffersink_get_time_base(sink->ctx);
        sink->frame_rate = av_buffersink_get_frame_rate(sink->ctx);
        sink->pts = AV_NOPTS_VALUE;
        sink->sample_index = 0;
        sink->samples = NULL;
        sink->nb_samples = 0;
        sink->render_ring_size = 2;
        ring_buffer_init(&sink->consume_frames);
        ring_buffer_init(&sink->render_frames);
        ring_buffer_init(&sink->purge_frames);

        glGenTextures(1, &sink->texture);

        std::thread sink_thread(worker_thread, &buffer_sinks[i], &mutexes[i], &cv[i]);

        video_sink_threads[i].swap(sink_thread);
    }

    std::vector<std::condition_variable> acv_list(abuffer_sinks.size());
    acv.swap(acv_list);

    std::vector<std::mutex> amutex_list(abuffer_sinks.size());
    amutexes.swap(amutex_list);

    std::vector<std::thread> athread_list(abuffer_sinks.size());
    audio_sink_threads.swap(athread_list);

    for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
        BufferSink *sink = &abuffer_sinks[i];

        sink->id = i;
        sink->label = av_asprintf("Audio FilterGraph Output %d", i);
        sink->time_base = av_buffersink_get_time_base(sink->ctx);
        sink->frame_rate = av_make_q(av_buffersink_get_sample_rate(sink->ctx), 1);
        sink->sample_index = 0;
        sink->nb_samples = 512;
        sink->pts = AV_NOPTS_VALUE;
        sink->samples = (float *)av_calloc(sink->nb_samples, sizeof(float));
        sink->render_ring_size = 2;
        ring_buffer_init(&sink->consume_frames);
        ring_buffer_init(&sink->render_frames);
        ring_buffer_init(&sink->purge_frames);

        sink->format = AL_FORMAT_MONO_FLOAT32;

        alGenBuffers(AL_BUFFERS, sink->buffers);
        for (unsigned j = 0; j < AL_BUFFERS; j++)
            sink->unprocessed_bufids.push_back(sink->buffers[j]);

        alGenSources(1, &sink->source);
        play_sources.push_back(sink->source);
        sink->gain = 1.f;
        sink->position[0] =  0.f;
        sink->position[1] =  0.f;
        sink->position[2] = -1.f;
        alSource3f(sink->source, AL_POSITION, sink->position[0], sink->position[1], sink->position[2]);
        alSourcei(sink->source, AL_SOURCE_RELATIVE, AL_TRUE);
        alSourcei(sink->source, AL_ROLLOFF_FACTOR, 0);

        std::thread asink_thread(worker_thread, &abuffer_sinks[i], &amutexes[i], &acv[i]);

        audio_sink_threads[i].swap(asink_thread);
    }

    std::thread new_sound_thread(sound_thread, abuffer_sinks.size(), &play_sources);
    play_sound_thread.swap(new_sound_thread);

    return 0;
}

static void load_frame(GLuint *out_texture, int *width, int *height, AVFrame *frame,
                       BufferSink *sink)
{
    *width  = frame->width;
    *height = frame->height;

    glBindTexture(GL_TEXTURE_2D, *out_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, sink->downscale_interpolator);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, sink->upscale_interpolator);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, frame->linesize[0] / 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frame->width, frame->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame->data[0]);
}

static void draw_info(bool *p_open, FrameInfo *frame)
{
    const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration |
                                          ImGuiWindowFlags_AlwaysAutoResize |
                                          ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoNav |
                                          ImGuiWindowFlags_NoMouseInputs |
                                          ImGuiWindowFlags_NoFocusOnAppearing |
                                          ImGuiWindowFlags_NoMove;

    ImGui::SetNextWindowPos(ImVec2(display_w/2, display_h/2), 0, ImVec2(0.5, 0.5));
    ImGui::SetNextWindowBgAlpha(0.7f);
    ImGui::SetNextWindowFocus();

    if (!ImGui::Begin("##Info", p_open, window_flags)) {
        ImGui::End();
        return;
    }

    ImGui::Separator();
    if (frame->width && frame->height) {
        ImGui::Text("SIZE: %dx%d", frame->width, frame->height);
        ImGui::Separator();
        ImGui::Text("KEY FRAME: %d", frame->key_frame);
        ImGui::Separator();
        ImGui::Text("PICTURE TYPE: %c", av_get_picture_type_char(frame->pict_type));
        ImGui::Separator();
        ImGui::Text("SAR: %dx%d", frame->sample_aspect_ratio.num, frame->sample_aspect_ratio.den);
        ImGui::Separator();
        ImGui::Text("COLOR RANGE: %s", av_color_range_name(frame->color_range));
        ImGui::Separator();
        ImGui::Text("COLOR PRIMARIES: %s", av_color_primaries_name(frame->color_primaries));
        ImGui::Separator();
        ImGui::Text("COLOR TRC: %s", av_color_transfer_name(frame->color_trc));
        ImGui::Separator();
        ImGui::Text("COLOR SPACE: %s", av_color_space_name(frame->colorspace));
        ImGui::Separator();
        ImGui::Text("CHROMA LOCATION: %s", av_chroma_location_name(frame->chroma_location));
        ImGui::Separator();
        ImGui::Text("PIXEL FORMAT: %s", av_get_pix_fmt_name((enum AVPixelFormat)frame->format));
        ImGui::Separator();
    } else if (frame->nb_samples) {
        ImGui::Text("SAMPLES: %d", frame->nb_samples);
        ImGui::Separator();
        ImGui::Text("SAMPLE RATE: %d", frame->sample_rate);
        ImGui::Separator();
        ImGui::Text("SAMPLE FORMAT: %s", av_get_sample_fmt_name((enum AVSampleFormat)frame->format));
        ImGui::Separator();
    }
    ImGui::Text("PTS: %ld", frame->pts);
    ImGui::Separator();
    ImGui::Text("TIME BASE: %d/%d", frame->time_base.num, frame->time_base.den);
    ImGui::Separator();
    ImGui::Text("PACKET POSITION: %ld", frame->pkt_pos);
    ImGui::Separator();
    ImGui::Text("PACKET SIZE: %d", frame->pkt_size);
    ImGui::Separator();
    ImGui::Text("PACKET DURATION: %ld", frame->pkt_duration);
    ImGui::End();
}

static void draw_help(bool *p_open)
{
    const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration |
                                          ImGuiWindowFlags_AlwaysAutoResize |
                                          ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoNav |
                                          ImGuiWindowFlags_NoMouseInputs |
                                          ImGuiWindowFlags_NoFocusOnAppearing |
                                          ImGuiWindowFlags_NoMove;
    const int align = 555;

    ImGui::SetNextWindowPos(ImVec2(display_w/2, display_h/2), 0, ImVec2(0.5, 0.5));
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::SetNextWindowFocus();

    if (!ImGui::Begin("##Help", p_open, window_flags)) {
        ImGui::End();
        return;
    }

    ImGui::Separator();
    ImGui::Separator();
    ImGui::Text("Global Keys:");
    ImGui::Separator();
    ImGui::Text("Show Help:");
    ImGui::SameLine(align);
    ImGui::Text("F1");
    ImGui::Separator();
    ImGui::Text("Jump to FilterGraph Editor:");
    ImGui::SameLine(align);
    ImGui::Text("F2");
    ImGui::Separator();
    ImGui::Text("Jump to FilterGraph Commands Window:");
    ImGui::SameLine(align);
    ImGui::Text("F3");
    ImGui::Separator();
    ImGui::Text("Jump to FilterGraph Dump Window:");
    ImGui::SameLine(align);
    ImGui::Text("F4");
    ImGui::Separator();
    ImGui::Text("Jump to FilterGraph Log Window:");
    ImGui::SameLine(align);
    ImGui::Text("F5");
    ImGui::Separator();
    ImGui::Text("Toggle Console:");
    ImGui::SameLine(align);
    ImGui::Text("Escape");
    ImGui::Separator();
    ImGui::Separator();
    ImGui::Separator();
    ImGui::Text("FilterGraph Editor Keys:");
    ImGui::Separator();
    ImGui::Text("Add New Filter:");
    ImGui::SameLine(align);
    ImGui::Text("A");
    ImGui::Separator();
    ImGui::Text("Auto Connect Filter Outputs to Sinks:");
    ImGui::SameLine(align);
    ImGui::Text("Shift + A");
    ImGui::Separator();
    ImGui::Text("Remove Selected Filters:");
    ImGui::SameLine(align);
    ImGui::Text("Shift + X");
    ImGui::Separator();
    ImGui::Text("Remove Selected Links:");
    ImGui::SameLine(align);
    ImGui::Text("X");
    ImGui::Separator();
    ImGui::Text("Clone Selected Filters:");
    ImGui::SameLine(align);
    ImGui::Text("Shift + C");
    ImGui::Separator();
    ImGui::Text("Configure Graph:");
    ImGui::SameLine(align);
    ImGui::Text("Ctrl + Enter");
    ImGui::Separator();
    ImGui::Separator();
    ImGui::Separator();
    ImGui::Text("Video/Audio FilterGraph Outputs:");
    ImGui::Separator();
    ImGui::Text("Pause playback:");
    ImGui::SameLine(align);
    ImGui::Text("Space");
    ImGui::Separator();
    ImGui::Text("Toggle fullscreen:");
    ImGui::SameLine(align);
    ImGui::Text("F");
    ImGui::Separator();
    ImGui::Text("Toggle zooming:");
    ImGui::SameLine(align);
    ImGui::Text("Z");
    ImGui::Separator();
    ImGui::Text("Framestep forward:");
    ImGui::SameLine(align);
    ImGui::Text("'.'");
    ImGui::Separator();
    ImGui::Text("Toggle OSD:");
    ImGui::SameLine(align);
    ImGui::Text("O");
    ImGui::Separator();
    ImGui::Text("Jump to #numbered Video output:");
    ImGui::SameLine(align);
    ImGui::Text("Ctrl + <number>");
    ImGui::Separator();
    ImGui::Text("Jump to #numbered Audio output:");
    ImGui::SameLine(align);
    ImGui::Text("Alt + <number>");
    ImGui::Separator();
    ImGui::Text("Toggle Audio output mute:");
    ImGui::SameLine(align);
    ImGui::Text("M");
    ImGui::Separator();
    ImGui::Text("Show Extended Info:");
    ImGui::SameLine(align);
    ImGui::Text("I");
    ImGui::Separator();
    ImGui::Text("Exit from output:");
    ImGui::SameLine(align);
    ImGui::Text("Shift + Q");
    ImGui::Separator();
    ImGui::End();
}

static void add_filter_node(const AVFilter *filter, ImVec2 pos)
{
    FilterNode node;

    node.edge = editor_edge++;
    node.filter = filter;
    node.id = filter_nodes.size();
    node.filter_name = av_strdup(filter->name);
    node.filter_label = av_asprintf("%s%d", filter->name, node.id);
    node.filter_options = NULL;
    node.ctx_options = NULL;
    node.probe = NULL;
    node.ctx = NULL;
    node.pos = pos;
    node.colapsed = false;
    node.set_pos = true;

    filter_nodes.push_back(node);
}

static void draw_console(bool *p_open)
{
    char input_line[4096] = { 0 };
    const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration |
                                          ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoNav |
                                          ImGuiWindowFlags_NoMove;

    ImGui::SetNextWindowPos(ImVec2(0, display_h - 35));
    ImGui::SetNextWindowSize(ImVec2(display_w, 30));
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::SetNextWindowFocus();

    if (!ImGui::Begin("##Console", p_open, window_flags)) {
        ImGui::End();
        return;
    }

    ImGuiInputTextFlags input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue;

    ImGui::SetKeyboardFocusHere();
    ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0,   0, 0, 200));
    ImGui::PushStyleColor(ImGuiCol_Text,    IM_COL32(0, 255, 0, 200));
    if (ImGui::InputText("##>", input_line, IM_ARRAYSIZE(input_line), input_text_flags)) {
        if (!strncmp(input_line, "a ", 2) && filter_graph_is_valid == false) {
            const AVFilter *filter = avfilter_get_by_name(input_line + 2);

            if (filter)
                add_filter_node(filter, ImVec2(0, 0));
        }
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();

    ImGui::End();
}

static void draw_osd(BufferSink *sink, int width, int height, int64_t pos)
{
    char osd_text[1024];

    snprintf(osd_text, sizeof(osd_text), "FRAME: %ld | SIZE: %dx%d | TIME: %.5f | SPEED: %011.5f | FPS: %d/%d (%.5f) | POS: %ld",
             sink->frame_number - 1,
             width, height,
             av_q2d(sink->time_base) * sink->pts,
             sink->speed,
             sink->frame_rate.num, sink->frame_rate.den, av_q2d(sink->frame_rate), pos);

    if (sink->fullscreen) {
        ImGui::SetCursorPos(ImVec2(20, 20));
        ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetCursorPos(),
                                                  ImVec2(ImGui::CalcTextSize(osd_text).x + 25, ImGui::GetFontSize() * 2.8f),
                                                  ImGui::GetColorU32(ImGuiCol_WindowBg, 0.5f));
    }

    ImGui::TextWrapped(osd_text);
}

static void update_frame_info(FrameInfo *frame_info, const AVFrame *frame)
{
    if (!ImGui::IsKeyDown(ImGuiKey_I))
        return;

    frame_info->width = frame->width;
    frame_info->height = frame->height;
    frame_info->nb_samples = frame->nb_samples;
    frame_info->format = frame->format;
    frame_info->key_frame = frame->key_frame;
    frame_info->pict_type = frame->pict_type;
    frame_info->sample_aspect_ratio = frame->sample_aspect_ratio;
    frame_info->pts = frame->pts;
    frame_info->pkt_dts = frame->pkt_dts;
    frame_info->time_base = frame->time_base;
    frame_info->coded_picture_number = frame->coded_picture_number;
    frame_info->display_picture_number = frame->display_picture_number;
    frame_info->interlaced_frame = frame->interlaced_frame;
    frame_info->top_field_first = frame->top_field_first;
    frame_info->sample_rate = frame->sample_rate;
    frame_info->color_range = frame->color_range;
    frame_info->color_primaries = frame->color_primaries;
    frame_info->color_trc = frame->color_trc;
    frame_info->colorspace = frame->colorspace;
    frame_info->chroma_location = frame->chroma_location;
    frame_info->pkt_pos = frame->pkt_pos;
    frame_info->pkt_duration = frame->pkt_duration;
    frame_info->pkt_size = frame->pkt_size;
    frame_info->crop_top = frame->crop_top;
    frame_info->crop_bottom = frame->crop_bottom;
    frame_info->crop_left = frame->crop_left;
    frame_info->crop_right = frame->crop_right;
}

static void draw_frame(GLuint *texture, bool *p_open, AVFrame *new_frame,
                       BufferSink *sink)
{
    ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
    int width, height;
    bool style = false;

    if (!*p_open || !new_frame)
        goto end;

    update_frame_info(&frame_info, new_frame);
    sink->pts = new_frame->pts;

    load_frame(texture, &width, &height, new_frame, sink);
    if (sink->fullscreen) {
        const ImGuiViewport *viewport = ImGui::GetMainViewport();

        sink->have_window_pos = true;

        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);

        flags |= ImGuiWindowFlags_NoDecoration;
        flags |= ImGuiWindowFlags_NoTitleBar;
        flags |= ImGuiWindowFlags_NoMove;
        flags |= ImGuiWindowFlags_NoResize;

        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        style = true;
    } else {
        ImGui::SetNextWindowSizeConstraints(ImVec2(width + 20, height), ImVec2(width + 20, height + 200));
        if (sink->have_window_pos == true) {
            ImGui::SetNextWindowPos(sink->window_pos);
            sink->have_window_pos = false;
        }
    }

    if (focus_buffersink_window == sink->id) {
        ImGui::SetNextWindowFocus();
        focus_buffersink_window = -1;
    }

    if (!ImGui::Begin(sink->label, p_open, flags)) {
        goto end;
    }

    if (sink->fullscreen == false)
        sink->window_pos = ImGui::GetWindowPos();

    if (ImGui::IsWindowFocused()) {
        if (ImGui::IsKeyReleased(ImGuiKey_F))
            sink->fullscreen = !sink->fullscreen;
        if (ImGui::IsKeyReleased(ImGuiKey_Space))
            paused = !paused;
        framestep = ImGui::IsKeyPressed(ImGuiKey_Period, true);
        if (framestep)
            paused = true;
        if (ImGui::IsKeyDown(ImGuiKey_Q) && ImGui::GetIO().KeyShift) {
            show_abuffersink_window = false;
            show_buffersink_window = false;
            filter_graph_is_valid = false;
        }
        if (ImGui::IsKeyReleased(ImGuiKey_O))
            sink->show_osd = !sink->show_osd;
    }

    if (ImGui::IsKeyDown(ImGuiKey_0 + sink->id) && ImGui::GetIO().KeyCtrl)
        focus_buffersink_window = sink->id;

    if (sink->fullscreen) {
        ImGui::GetWindowDrawList()->AddImage((void*)(intptr_t)*texture, ImVec2(0.f, 0.f),
                                             ImGui::GetWindowSize(),
                                             ImVec2(0.f, 0.f), ImVec2(1.f, 1.f), IM_COL32_WHITE);
    } else {
        ImGui::Image((void*)(intptr_t)*texture, ImVec2(width, height));
    }

    if ((ImGui::IsItemHovered() || sink->fullscreen) && ImGui::IsKeyDown(ImGuiKey_Z)) {
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 size = ImGui::GetWindowSize();
        ImVec2 pos = ImGui::GetWindowPos();
        ImGui::BeginTooltip();
        float my_tex_w = (float)width;
        float my_tex_h = (float)height;
        ImVec4 tint_col   = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // No tint
        ImVec4 border_col = ImVec4(1.0f, 1.0f, 1.0f, 0.5f); // 50% opaque white
        float region_sz = 32.0f;
        float region_x = (io.MousePos.x - pos.x - region_sz * 0.5f) * width  / size.x;
        float region_y = (io.MousePos.y - pos.y - region_sz * 0.5f) * height / size.y;
        static float zoom = 4.f;
        zoom = av_clipf(zoom + io.MouseWheel * 0.3f, 1.5f, 12.f);
        if (region_x < 0.0f) { region_x = 0.0f; }
        else if (region_x > my_tex_w - region_sz) { region_x = my_tex_w - region_sz; }
        if (region_y < 0.0f) { region_y = 0.0f; }
        else if (region_y > my_tex_h - region_sz) { region_y = my_tex_h - region_sz; }
        ImVec2 uv0 = ImVec2((region_x) / my_tex_w, (region_y) / my_tex_h);
        ImVec2 uv1 = ImVec2((region_x + region_sz) / my_tex_w, (region_y + region_sz) / my_tex_h);
        ImGui::Image((void*)(intptr_t)*texture, ImVec2(region_sz * zoom, region_sz * zoom), uv0, uv1, tint_col, border_col);
        ImGui::EndTooltip();
    }

    if (sink->show_osd)
        draw_osd(sink, width, height, new_frame->pkt_pos);

    if (style) {
        ImGui::PopStyleVar();
        ImGui::PopStyleVar();
    }

    ImGui::End();

end:

    if (new_frame && new_frame->nb_samples == 0) {
        sink->frame_number++;
        new_frame->nb_samples = 1;
    }
}

static void draw_aframe(bool *p_open, BufferSink *sink)
{
    ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
    ALint queued;

    if (!*p_open)
        return;

    if (focus_abuffersink_window == sink->id) {
        ImGui::SetNextWindowFocus();
        focus_abuffersink_window = -1;
    }

    if (!ImGui::Begin(sink->label, p_open, flags)) {
        ImGui::End();
        return;
    }

    if (ImGui::IsWindowFocused()) {
        if (ImGui::IsKeyReleased(ImGuiKey_Space))
            paused = !paused;
        if (ImGui::IsKeyReleased(ImGuiKey_M))
            sink->muted = !sink->muted;
        framestep = ImGui::IsKeyPressed(ImGuiKey_Period, true);
        if (framestep)
            paused = true;
        if (ImGui::IsKeyDown(ImGuiKey_Q) && ImGui::GetIO().KeyShift) {
            show_abuffersink_window = false;
            show_buffersink_window = false;
            filter_graph_is_valid = false;
        }
        if (ImGui::IsKeyReleased(ImGuiKey_O))
            sink->show_osd = !sink->show_osd;
    }

    if (ImGui::IsKeyDown(ImGuiKey_0 + sink->id) && ImGui::GetIO().KeyAlt)
        focus_abuffersink_window = sink->id;

    ImVec2 window_size = { audio_window_size[0], audio_window_size[1] };
    ImGui::PlotLines("##Audio Samples", sink->samples, sink->nb_samples, 0, NULL, -audio_sample_range[0], audio_sample_range[1], window_size);
    if (sink->show_osd) {
        ImGui::Text("FRAME: %ld", sink->frame_number);
        ImGui::Text("SIZE:  %d", sink->frame_nb_samples);
        ImGui::Text("TIME:  %.5f", sink->pts != AV_NOPTS_VALUE ? av_q2d(sink->time_base) * sink->pts : NAN);
        ImGui::Text("SPEED: %011.5f", sink->speed);
        alGetSourcei(sink->source, AL_BUFFERS_QUEUED, &queued);
        ImGui::Text("POS:   %ld", sink->pos);
        ImGui::Text("QUEUE: %d", queued);
    }
    if (ImGui::DragFloat("Gain", &sink->gain, 0.01f, 0.f, 2.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput))
        alSourcef(sink->source, AL_GAIN, sink->gain);
    if (ImGui::DragFloat3("Position", sink->position, 0.01f, -1.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput))
        alSource3f(sink->source, AL_POSITION, sink->position[0], sink->position[1], sink->position[2]);

    ImGui::End();
}

static void queue_sound(AVFrame *frame, BufferSink *sink)
{
    ALint processed = 0;

    alSourcef(sink->source, AL_GAIN, sink->gain * !sink->muted);

    alGetSourcei(sink->source, AL_BUFFERS_PROCESSED, &processed);
    while (processed > 0 && (!paused || framestep)) {
        ALuint bufid;

        alSourceUnqueueBuffers(sink->source, 1, &bufid);
        processed--;

        sink->processed_bufids.push_back(bufid);
    }

    if (sink->processed_bufids.size() > 0 && frame->nb_samples > 0) {
        ALuint bufid = sink->processed_bufids.back();

        sink->processed_bufids.pop_back();
        alBufferData(bufid, sink->format, frame->extended_data[0],
                     (ALsizei)frame->nb_samples * sizeof(float), frame->sample_rate);
        alSourceQueueBuffers(sink->source, 1, &bufid);
        frame->nb_samples = 0;
    } else if (sink->unprocessed_bufids.size() > 0 && frame->nb_samples > 0) {
        ALuint bufid = sink->unprocessed_bufids.back();

        sink->unprocessed_bufids.pop_back();
        alBufferData(bufid, sink->format, frame->extended_data[0],
                     (ALsizei)frame->nb_samples * sizeof(float), frame->sample_rate);
        alSourceQueueBuffers(sink->source, 1, &bufid);
        frame->nb_samples = 0;
    }
}

static bool query_ranges(void *obj, const AVOption *opt,
                         double *min, double *max)
{
    switch (opt->type) {
        case AV_OPT_TYPE_INT:
        case AV_OPT_TYPE_INT64:
        case AV_OPT_TYPE_UINT64:
        case AV_OPT_TYPE_DOUBLE:
        case AV_OPT_TYPE_FLOAT:
        case AV_OPT_TYPE_RATIONAL:
        case AV_OPT_TYPE_BOOL:
        case AV_OPT_TYPE_FLAGS:
        case AV_OPT_TYPE_DURATION:
            *min = opt->min;
            *max = opt->max;
            break;
        default:
            break;
    }

    return true;
}

static bool is_simple_filter(const AVFilter *filter)
{
    if (avfilter_filter_pad_count(filter, 0) == 1 &&
        avfilter_filter_pad_count(filter, 1) == 1)
        return true;
    return false;
}

static bool is_simple_audio_filter(const AVFilter *filter)
{
    if (is_simple_filter(filter) &&
        avfilter_pad_get_type(filter->inputs,  0) == AVMEDIA_TYPE_AUDIO &&
        avfilter_pad_get_type(filter->outputs, 0) == AVMEDIA_TYPE_AUDIO) {
        return true;
    }
    return false;
}

static bool is_simple_video_filter(const AVFilter *filter)
{
    if (is_simple_filter(filter) &&
        avfilter_pad_get_type(filter->inputs,  0) == AVMEDIA_TYPE_VIDEO &&
        avfilter_pad_get_type(filter->outputs, 0) == AVMEDIA_TYPE_VIDEO) {
        return true;
    }
    return false;
}

static bool is_source_filter(const AVFilter *filter)
{
    if ((avfilter_filter_pad_count(filter, 0)  > 0  ||  (filter->flags & AVFILTER_FLAG_DYNAMIC_INPUTS)) ||
        (avfilter_filter_pad_count(filter, 1) == 0  && !(filter->flags & AVFILTER_FLAG_DYNAMIC_OUTPUTS))) {
        return false;
    }

    return true;
}

static bool is_sink_filter(const AVFilter *filter)
{
    if ((avfilter_filter_pad_count(filter, 0)  > 0  ||  (filter->flags & AVFILTER_FLAG_DYNAMIC_INPUTS)) &&
        (avfilter_filter_pad_count(filter, 1) == 0  && !(filter->flags & AVFILTER_FLAG_DYNAMIC_OUTPUTS))) {
        return true;
    }

    return false;
}

static bool is_source_audio_filter(const AVFilter *filter)
{
    if (is_source_filter(filter)) {
        for (unsigned i = 0; i < avfilter_filter_pad_count(filter, 1); i++) {
            if (avfilter_pad_get_type(filter->outputs, i) != AVMEDIA_TYPE_AUDIO)
                return false;
        }

        return true;
    }

    return false;
}

static bool is_source_video_filter(const AVFilter *filter)
{
    if (is_source_filter(filter)) {
        for (unsigned i = 0; i < avfilter_filter_pad_count(filter, 1); i++) {
            if (avfilter_pad_get_type(filter->outputs, i) != AVMEDIA_TYPE_VIDEO)
                return false;
        }

        return true;
    }

    return false;
}

static bool is_source_media_filter(const AVFilter *filter)
{
    if (is_source_filter(filter) && !is_source_audio_filter(filter) && !is_source_video_filter(filter))
        return true;
    return false;
}

static bool is_sink_audio_filter(const AVFilter *filter)
{
    if (is_sink_filter(filter)) {
        for (unsigned i = 0; i < avfilter_filter_pad_count(filter, 0); i++) {
            if (avfilter_pad_get_type(filter->inputs, i) != AVMEDIA_TYPE_AUDIO)
                return false;
        }

        return true;
    }

    return false;
}

static bool is_sink_video_filter(const AVFilter *filter)
{
    if (is_sink_filter(filter)) {
        for (unsigned i = 0; i < avfilter_filter_pad_count(filter, 0); i++) {
            if (avfilter_pad_get_type(filter->inputs, i) != AVMEDIA_TYPE_VIDEO)
                return false;
        }

        return true;
    }

    return false;
}

static bool is_media_filter(const AVFilter *filter)
{
    if (is_simple_filter(filter) &&
        avfilter_pad_get_type(filter->inputs, 0) !=
        avfilter_pad_get_type(filter->outputs, 0)) {
        return true;
    }

    return false;
}

static bool is_complex_filter(const AVFilter *filter)
{
    if (!is_sink_filter(filter) && !is_source_filter(filter) && !is_simple_filter(filter))
        return true;
    return false;
}

static void handle_nodeitem(const AVFilter *filter, ImVec2 click_pos)
{
    if (ImGui::MenuItem(filter->name))
        add_filter_node(filter, click_pos);

    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", filter->description);
}

static void draw_options(FilterNode *node, void *av_class)
{
    const AVOption *opt = NULL;
    const void *obj = av_class;
    int last_offset = -1;
    double min, max;
    int index = 0;

    while ((opt = av_opt_next(obj, opt))) {
        if (last_offset == opt->offset)
            continue;
        last_offset = opt->offset;
        if (!query_ranges((void *)obj, opt, &min, &max))
            continue;
        switch (opt->type) {
            case AV_OPT_TYPE_INT64:
                {
                    int64_t value;
                    int64_t smin = min;
                    int64_t smax = max;
                    if (av_opt_get_int(av_class, opt->name, 0, &value))
                        break;
                    if (ImGui::DragScalar(opt->name, ImGuiDataType_S64, &value, 1, &smin, &smax, "%ld", ImGuiSliderFlags_AlwaysClamp)) {
                        av_opt_set_int(av_class, opt->name, value, 0);
                    }
                }
                break;
            case AV_OPT_TYPE_UINT64:
                {
                    int64_t value;
                    uint64_t uvalue;
                    uint64_t umin = min;
                    uint64_t umax = max;
                    if (av_opt_get_int(av_class, opt->name, 0, &value))
                        break;
                    uvalue = value;
                    if (ImGui::DragScalar(opt->name, ImGuiDataType_U64, &value, 1, &umin, &umax, "%lu", ImGuiSliderFlags_AlwaysClamp)) {
                        value = uvalue;
                        av_opt_set_int(av_class, opt->name, value, 0);
                    }
                }
                break;
            case AV_OPT_TYPE_DURATION:
                {
                    int64_t value;
                    double dvalue;
                    if (av_opt_get_int(av_class, opt->name, 0, &value))
                        break;
                    dvalue = value / 1000000.0;
                    if (ImGui::DragScalar(opt->name, ImGuiDataType_Double, &dvalue, 0.1f, &min, &max, "%f", ImGuiSliderFlags_AlwaysClamp)) {
                        value = dvalue * 1000000.0;
                        av_opt_set_int(av_class, opt->name, value, 0);
                    }
                }
                break;
            case AV_OPT_TYPE_FLAGS:
            case AV_OPT_TYPE_BOOL:
                {
                    int64_t value;
                    int ivalue;
                    int imin = min;
                    int imax = max;
                    if (av_opt_get_int(av_class, opt->name, 0, &value))
                        break;
                    ivalue = value;
                    if (imax < INT_MAX/2 && imin > INT_MIN/2) {
                        if (ImGui::SliderInt(opt->name, &ivalue, imin, imax)) {
                            value = ivalue;
                            av_opt_set_int(av_class, opt->name, value, 0);
                        }
                    } else {
                        if (ImGui::DragInt(opt->name, &ivalue, imin, imax, ImGuiSliderFlags_AlwaysClamp)) {
                            value = ivalue;
                            av_opt_set_int(av_class, opt->name, value, 0);
                        }
                    }

                    if (opt->unit) {
                        char combo_name[20];

                        snprintf(combo_name, sizeof(combo_name), "##%s", opt->unit);
                        if (ImGui::BeginCombo(combo_name, 0, 0)) {
                            const AVOption *copt = NULL;

                            while ((copt = av_opt_next(obj, copt))) {
                                const bool is_selected = value == copt->default_val.i64;

                                if (!copt->unit)
                                    continue;
                                if (strcmp(copt->unit, opt->unit) || copt->type != AV_OPT_TYPE_CONST)
                                    continue;

                                if (ImGui::Selectable(copt->name, is_selected))
                                    av_opt_set_int(av_class, opt->name, copt->default_val.i64, 0);
                                ImGui::SameLine();
                                ImGui::Text("\t\t%s", copt->help);

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }

                            ImGui::EndCombo();
                        }
                    }
                }
                break;
            case AV_OPT_TYPE_INT:
                {
                    int64_t value;
                    int ivalue;
                    int imin = min;
                    int imax = max;
                    if (av_opt_get_int(av_class, opt->name, 0, &value))
                        break;
                    ivalue = value;
                    if (imax < INT_MAX/2 && imin > INT_MIN/2) {
                        if (ImGui::SliderInt(opt->name, &ivalue, imin, imax)) {
                            value = ivalue;
                            av_opt_set_int(av_class, opt->name, value, 0);
                        }
                    } else {
                        if (ImGui::DragInt(opt->name, &ivalue, imin, imax, ImGuiSliderFlags_AlwaysClamp)) {
                            value = ivalue;
                            av_opt_set_int(av_class, opt->name, value, 0);
                        }
                    }

                    if (opt->unit) {
                        char combo_name[20];

                        snprintf(combo_name, sizeof(combo_name), "##%s", opt->unit);
                        if (ImGui::BeginCombo(combo_name, 0, 0)) {
                            const AVOption *copt = NULL;

                            while ((copt = av_opt_next(obj, copt))) {
                                const bool is_selected = value == copt->default_val.i64;

                                if (!copt->unit)
                                    continue;
                                if (strcmp(copt->unit, opt->unit) || copt->type != AV_OPT_TYPE_CONST)
                                    continue;

                                if (ImGui::Selectable(copt->name, is_selected))
                                    av_opt_set_int(av_class, opt->name, copt->default_val.i64, 0);
                                ImGui::SameLine();
                                ImGui::Text("\t\t%s", copt->help);

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }

                            ImGui::EndCombo();
                        }
                    }
                }
                break;
            case AV_OPT_TYPE_DOUBLE:
                {
                    double value;

                    if (av_opt_get_double(av_class, opt->name, 0, &value))
                        break;
                    if (ImGui::DragScalar(opt->name, ImGuiDataType_Double, &value, 1.0, &min, &max, "%f", ImGuiSliderFlags_AlwaysClamp)) {
                        av_opt_set_double(av_class, opt->name, value, 0);
                    }
                }
                break;
            case AV_OPT_TYPE_FLOAT:
                {
                    double value;
                    float fvalue;
                    float fmin = min;
                    float fmax = max;

                    if (av_opt_get_double(av_class, opt->name, 0, &value))
                        break;
                    fvalue = value;
                    if (ImGui::DragFloat(opt->name, &fvalue, 1.f, fmin, fmax, "%f", ImGuiSliderFlags_AlwaysClamp)) {
                        value = fvalue;
                        av_opt_set_double(av_class, opt->name, value, 0);
                    }
                }
                break;
            case AV_OPT_TYPE_STRING:
                {
                    char new_str[1024] = {0};
                    uint8_t *str = NULL;

                    if (av_opt_get(av_class, opt->name, 0, &str))
                        break;
                    if (str) {
                        memcpy(new_str, str, strlen((const char *)str));
                        av_freep(&str);
                    }
                    if (ImGui::InputText(opt->name, new_str, IM_ARRAYSIZE(new_str))) {
                        av_opt_set(av_class, opt->name, new_str, 0);
                    }
                }
                break;
            case AV_OPT_TYPE_RATIONAL:
                {
                    AVRational rate = (AVRational){ 0, 0 };
                    int irate[2] = { 0, 0 };

                    if (rate.num == 0 && rate.den == 0)
                        av_opt_get_q(av_class, opt->name, 0, &rate);
                    irate[0] = rate.num;
                    irate[1] = rate.den;
                    if (ImGui::DragInt2(opt->name, irate, 1, -8192, 8192)) {
                        rate.num = irate[0];
                        rate.den = irate[1];
                        av_opt_set_q(av_class, opt->name, rate, 0);
                    }
                }
                break;
            case AV_OPT_TYPE_BINARY:
                break;
            case AV_OPT_TYPE_DICT:
                break;
            case AV_OPT_TYPE_IMAGE_SIZE:
                {
                    int size[2] = {0,0};

                    av_opt_get_image_size(av_class, opt->name, 0, &size[0], &size[1]);
                    if (ImGui::DragInt2(opt->name, size, 1, 1, 4096)) {
                        av_opt_set_image_size(av_class, opt->name, size[0], size[1], 0);
                    }
                }
                break;
            case AV_OPT_TYPE_VIDEO_RATE:
                {
                    int irate[2] = { 0, 0 };
                    uint8_t *old_rate_str = NULL;
                    char rate_str[256];

                    av_opt_get(av_class, opt->name, 0, &old_rate_str);
                    if (old_rate_str) {
                        sscanf((const char *)old_rate_str, "%d/%d", &irate[0], &irate[1]);
                        av_freep(&old_rate_str);
                    }
                    if (ImGui::DragInt2(opt->name, irate, 1, -8192, 8192)) {
                        snprintf(rate_str, sizeof(rate_str), "%d/%d", irate[0], irate[1]);
                        av_opt_set(av_class, opt->name, rate_str, 0);
                    }
                }
                break;
            case AV_OPT_TYPE_PIXEL_FMT:
                if (ImGui::BeginCombo("pixel format", 0, 0)) {
                    const AVPixFmtDescriptor *pix_desc = NULL;
                    AVPixelFormat fmt;

                    av_opt_get_pixel_fmt(av_class, opt->name, 0, &fmt);
                    while ((pix_desc = av_pix_fmt_desc_next(pix_desc))) {
                        enum AVPixelFormat pix_fmt = av_pix_fmt_desc_get_id(pix_desc);
                        const bool is_selected = av_pix_fmt_desc_get_id(pix_desc) == fmt;

                        if (ImGui::Selectable(pix_desc->name, is_selected))
                            av_opt_set_pixel_fmt(av_class, opt->name, pix_fmt, 0);

                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                break;
            case AV_OPT_TYPE_SAMPLE_FMT:
                break;
            case AV_OPT_TYPE_COLOR:
                {
                    float col[4] = { 0.4f, 0.7f, 0.0f, 0.5f };
                    unsigned icol[4] = { 0 };
                    char new_str[16] = { 0 };
                    uint8_t *old_str = NULL;

                    if (av_opt_get(av_class, opt->name, 0, &old_str))
                        break;
                    sscanf((const char *)old_str, "0x%02x%02x%02x%02X", &icol[0], &icol[1], &icol[2], &icol[3]);
                    av_freep(&old_str);
                    col[0] = icol[0] / 255.f;
                    col[1] = icol[1] / 255.f;
                    col[2] = icol[2] / 255.f;
                    col[3] = icol[3] / 255.f;
                    ImGui::PushID(index++);
                    ImGui::ColorEdit4("color", col, ImGuiColorEditFlags_NoDragDrop);
                    ImGui::PopID();
                    icol[0] = col[0] * 255.f;
                    icol[1] = col[1] * 255.f;
                    icol[2] = col[2] * 255.f;
                    icol[3] = col[3] * 255.f;
                    snprintf(new_str, sizeof(new_str), "0x%02x%02x%02x%02x", icol[0], icol[1], icol[2], icol[3]);
                    av_opt_set(av_class, opt->name, new_str, 0);
                }
                break;
            case AV_OPT_TYPE_CHLAYOUT:
                break;
            case AV_OPT_TYPE_CONST:
                break;
            default:
                break;
        }

        if (ImGui::IsItemHovered() && opt->type != AV_OPT_TYPE_CONST)
            ImGui::SetTooltip("%s", opt->help);
    }
}

static void draw_filter_commands(const AVFilterContext *ctx, unsigned n, unsigned *toggle_filter,
                                 bool is_opened, bool *clean_storage, bool tree)
{
    if (ctx->filter->process_command) {
        if (tree == false) {
            if (filter_nodes[n].colapsed == false && !ImGui::Button("Commands"))
                return;
            filter_nodes[n].colapsed = true;
            if (filter_nodes[n].colapsed == true && ImGui::Button("Close"))
                filter_nodes[n].colapsed = false;
        }

        if (tree ? ImGui::TreeNode("Commands") : ImGui::BeginListBox("##Commands", ImVec2(200, 100))) {
            std::vector<OptStorage> opt_storage = filter_nodes[n].opt_storage;
            const AVOption *opt = NULL;
            unsigned opt_index = 0;

            if (is_opened && *clean_storage) {
                opt_storage.clear();
                *clean_storage = false;
            }

            while ((opt = av_opt_next(ctx->priv, opt))) {
                double min, max;
                void *ptr;

                if (!(opt->flags & AV_OPT_FLAG_RUNTIME_PARAM))
                    continue;

                if (!query_ranges((void *)&ctx->filter->priv_class, opt, &min, &max))
                    continue;

                ptr = av_opt_ptr(ctx->filter->priv_class, ctx->priv, opt->name);
                if (!ptr)
                    continue;

                ImGui::PushID(opt_index);
                switch (opt->type) {
                    case AV_OPT_TYPE_FLAGS:
                    case AV_OPT_TYPE_BOOL:
                    case AV_OPT_TYPE_INT:
                    case AV_OPT_TYPE_DOUBLE:
                    case AV_OPT_TYPE_FLOAT:
                    case AV_OPT_TYPE_INT64:
                    case AV_OPT_TYPE_UINT64:
                    case AV_OPT_TYPE_STRING:
                        if (ImGui::Button("Send")) {
                            char arg[1024] = { 0 };

                            switch (opt->type) {
                                case AV_OPT_TYPE_FLAGS:
                                case AV_OPT_TYPE_BOOL:
                                case AV_OPT_TYPE_INT:
                                    snprintf(arg, sizeof(arg) - 1, "%d", opt_storage[opt_index].u.i32);
                                    break;
                                case AV_OPT_TYPE_INT64:
                                    snprintf(arg, sizeof(arg) - 1, "%ld", opt_storage[opt_index].u.i64);
                                    break;
                                case AV_OPT_TYPE_UINT64:
                                    snprintf(arg, sizeof(arg) - 1, "%lu", opt_storage[opt_index].u.u64);
                                    break;
                                case AV_OPT_TYPE_DOUBLE:
                                    snprintf(arg, sizeof(arg) - 1, "%f", opt_storage[opt_index].u.dbl);
                                    break;
                                case AV_OPT_TYPE_FLOAT:
                                    snprintf(arg, sizeof(arg) - 1, "%f", opt_storage[opt_index].u.flt);
                                    break;
                                case AV_OPT_TYPE_STRING:
                                    snprintf(arg, strlen(opt_storage[opt_index].u.str) + 1, "%s", opt_storage[opt_index].u.str);
                                    break;
                                default:
                                    break;
                            }

                            avfilter_graph_send_command(filter_graph, ctx->name, opt->name, arg, NULL, 0, 0);
                        }
                        ImGui::SameLine();
                    default:
                        break;
                }

                switch (opt->type) {
                    case AV_OPT_TYPE_FLAGS:
                    case AV_OPT_TYPE_BOOL:
                        {
                            int value = *(int *)ptr;
                            int imin = min;
                            int imax = max;

                            if (opt_storage.size() <= opt_index) {
                                OptStorage new_opt;

                                new_opt.u.i32 = *(int *)ptr;
                                opt_storage.push_back(new_opt);
                            }

                            value = opt_storage[opt_index].u.i32;
                            if (ImGui::SliderInt(opt->name, &value, imin, imax)) {
                                opt_storage[opt_index].u.i32 = value;
                            }
                        }
                        break;
                    case AV_OPT_TYPE_INT:
                        {
                            int value = *(int *)ptr;
                            int imin = min;
                            int imax = max;

                            if (opt_storage.size() <= opt_index) {
                                OptStorage new_opt;

                                new_opt.u.i32 = *(int *)ptr;
                                opt_storage.push_back(new_opt);
                            }
                            value = opt_storage[opt_index].u.i32;
                            if (imax < INT_MAX/2 && imin > INT_MIN/2) {
                                if (ImGui::SliderInt(opt->name, &value, imin, imax)) {
                                    opt_storage[opt_index].u.i32 = value;
                                }
                            } else {
                                if (ImGui::DragInt(opt->name, &value, imin, imax, ImGuiSliderFlags_AlwaysClamp)) {
                                    opt_storage[opt_index].u.i32 = value;
                                }
                            }
                        }
                        break;
                    case AV_OPT_TYPE_INT64:
                        {
                            int64_t value = *(int64_t *)ptr;
                            int64_t imin = min;
                            int64_t imax = max;

                            if (opt_storage.size() <= opt_index) {
                                OptStorage new_opt;

                                new_opt.u.i64 = *(int64_t *)ptr;
                                opt_storage.push_back(new_opt);
                            }
                            value = opt_storage[opt_index].u.i64;
                            if (ImGui::DragScalar(opt->name, ImGuiDataType_S64, &value, 1, &imin, &imax, "%ld", ImGuiSliderFlags_AlwaysClamp)) {
                                opt_storage[opt_index].u.i64 = value;
                            }
                        }
                        break;
                    case AV_OPT_TYPE_UINT64:
                        {
                            uint64_t value = *(uint64_t *)ptr;
                            uint64_t umin = min;
                            uint64_t umax = max;

                            if (opt_storage.size() <= opt_index) {
                                OptStorage new_opt;

                                new_opt.u.u64 = *(uint64_t *)ptr;
                                opt_storage.push_back(new_opt);
                            }
                            value = opt_storage[opt_index].u.u64;
                            if (ImGui::DragScalar(opt->name, ImGuiDataType_U64, &value, 1, &umin, &umax, "%lu", ImGuiSliderFlags_AlwaysClamp)) {
                                opt_storage[opt_index].u.u64 = value;
                            }
                        }
                        break;
                    case AV_OPT_TYPE_DOUBLE:
                        {
                            double value = *(double *)ptr;

                            if (opt_storage.size() <= opt_index) {
                                OptStorage new_opt;

                                new_opt.u.dbl = *(double *)ptr;
                                opt_storage.push_back(new_opt);
                            }
                            value = opt_storage[opt_index].u.dbl;
                            if (ImGui::DragScalar(opt->name, ImGuiDataType_Double, &value, 1.0, &min, &max, "%f", ImGuiSliderFlags_AlwaysClamp)) {
                                opt_storage[opt_index].u.dbl = value;
                            }
                        }
                        break;
                    case AV_OPT_TYPE_FLOAT:
                        {
                            float fmax = max;
                            float fmin = min;
                            float value;

                            if (opt_storage.size() <= opt_index) {
                                OptStorage new_opt;

                                new_opt.u.flt = *(float *)ptr;
                                opt_storage.push_back(new_opt);
                            }
                            value = opt_storage[opt_index].u.flt;
                            if (ImGui::DragFloat(opt->name, &value, 1.f, fmin, fmax, "%f", ImGuiSliderFlags_AlwaysClamp))
                                opt_storage[opt_index].u.flt = value;
                        }
                        break;
                    case AV_OPT_TYPE_STRING:
                        {
                            char string[1024] = { 0 };
                            uint8_t *str = NULL;

                            if (opt_storage.size() <= opt_index) {
                                OptStorage new_opt;

                                av_opt_get(ctx->priv, opt->name, 0, &str);
                                new_opt.u.str = (char *)str;
                                opt_storage.push_back(new_opt);
                            }

                            if (opt_storage[opt_index].u.str)
                                memcpy(string, opt_storage[opt_index].u.str, std::min(sizeof(string) - 1, strlen(opt_storage[opt_index].u.str)));
                            if (ImGui::InputText(opt->name, string, sizeof(string) - 1)) {
                                av_freep(&opt_storage[opt_index].u.str);
                                opt_storage[opt_index].u.str = av_strdup(string);
                            }
                        }
                        break;
                    default:
                        break;
                }

                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", opt->help);

                opt_index++;

                ImGui::PopID();
            }

            filter_nodes[n].opt_storage = opt_storage;

            tree ? ImGui::TreePop() : ImGui::EndListBox();
        }
    }

    if (ctx->filter->flags & AVFILTER_FLAG_SUPPORT_TIMELINE) {
        if (tree ? ImGui::TreeNode("Timeline") : ImGui::BeginListBox("##Timeline", ImVec2(80, 30))) {
            ImGui::PushID(0);
            if (ImGui::Button(ctx->is_disabled ? "Enable" : "Disable"))
                *toggle_filter = n;
            ImGui::PopID();
            tree ? ImGui::TreePop() : ImGui::EndListBox();
        }
    }
}

static void draw_node_options(FilterNode *node)
{
    AVFilterContext *probe_ctx;
    void *av_class_priv;
    void *av_class;

    if (!probe_graph)
        probe_graph = avfilter_graph_alloc();
    if (!probe_graph)
        return;
    probe_graph->nb_threads = 1;

    if (!node->probe)
        node->probe = avfilter_graph_alloc_filter(probe_graph, node->filter, "probe");
    probe_ctx = node->probe;
    if (!probe_ctx)
        return;

    if (filter_graph_is_valid) {
        static unsigned toggle_filter = INT_MAX;
        static bool clean_storage = true;

        draw_filter_commands(node->ctx, node->id, &toggle_filter, true, &clean_storage, false);

        if (toggle_filter < UINT_MAX) {
            const AVFilterContext *filter_ctx = node->ctx;
            const int flag = !filter_ctx->is_disabled;

            avfilter_graph_send_command(filter_graph, filter_ctx->name, "enable", flag ? "0" : "1", NULL, 0, 0);
            toggle_filter = UINT_MAX;
        }

        return;
    }

    av_class_priv = probe_ctx->priv;
    av_class = probe_ctx;
    if (!node->colapsed && !ImGui::Button("Options"))
        return;

    node->colapsed = true;
    if (node->colapsed && ImGui::Button("Close")) {
        node->colapsed = false;
        return;
    }

    ImGui::SameLine();
    if (node->colapsed) {
        for (unsigned i = 0; i < video_sink_threads.size(); i++) {
            if (video_sink_threads[i].joinable())
                return;
        }

        for (unsigned i = 0; i < audio_sink_threads.size(); i++) {
            if (audio_sink_threads[i].joinable())
                return;
        }

        if (ImGui::Button("Remove")) {
            av_freep(&node->filter_name);
            av_freep(&node->filter_label);
            av_freep(&node->filter_options);
            av_freep(&node->ctx_options);
            avfilter_free(node->probe);
            node->probe = NULL;
            avfilter_free(node->ctx);
            node->ctx = NULL;
            node->colapsed = false;
            edge2pad[node->edge].is_output = false;
            edge2pad[node->edge].removed = true;
            filter_nodes.erase(filter_nodes.begin() + node->id);
            return;
        }
    }

    if (!ImGui::BeginListBox("##List of Filter Options"))
        return;
    draw_options(node, av_class_priv);
    ImGui::NewLine();
    draw_options(node, av_class);

    ImGui::EndListBox();
}

static ImVec2 find_node_spot(ImVec2 start);

static void import_filter_graph(const char *file_name)
{
    FILE *file = av_fopen_utf8(file_name, "r");
    std::vector<char *> filter_opts;
    std::vector<std::pair <int, int>> labels;
    std::vector<std::pair <int, int>> filters;
    std::vector<std::pair <unsigned, unsigned>> pads;
    std::vector<int> separators;
    std::vector<int> filter2edge;
    std::vector<int> label2edge;
    int filter_start = -1;
    int filter_stop = 0;
    int label_start = 0;
    int label_stop = 0;
    int in_pad_count = -1;
    int out_pad_count = -1;
    int pos = 0;
    AVBPrint buf;
    char c;

    if (!file) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open '%s' script.\n", file_name);
        return;
    }

    if (!probe_graph)
        probe_graph = avfilter_graph_alloc();
    if (!probe_graph)
        return;

    av_bprint_init(&buf, 512, AV_BPRINT_SIZE_UNLIMITED);

    while ((c = fgetc(file))) {
        if (c != EOF)
            av_bprintf(&buf, "%c", c);

        if (c == ';' || c == EOF)
            separators.push_back(pos);

        if (c == '[' && label_start == 0 && label_stop == 0) {
            if (filter_start >= 0) {
                filter_stop = pos;
                if (filter_stop - filter_start > 1)
                    filters.push_back(std::make_pair(filter_start, filter_stop));
                filter_start = -1;
                filter_stop = 0;
            }
            label_start = pos + 1;
        } else if (c == ']' && label_start > 0 && label_stop == 0) {
            label_stop = pos;
            labels.push_back(std::make_pair(label_start, label_stop));
            label_start = label_stop = 0;
        } else if ((c == ';' || c == EOF) && filter_start >= 0) {
            filter_stop = pos - (c == EOF);
            if (filter_stop - filter_start > 1)
                filters.push_back(std::make_pair(filter_start, filter_stop));
            filter_start = -1;
            filter_stop = 0;
        } else if (filter_start >= 0) {
        } else if (label_stop == 0 && label_start == 0) {
            filter_start = pos + (c == ';');
        }

        pos++;
        if (c == EOF)
            break;
    }

    unsigned cur_label = 0;
    unsigned cur_filter_idx = 0;
    int label, filter, separator;

    if (separators.size() != filters.size())
        goto error;

    while (cur_label < labels.size()) {
        label = labels[cur_label].second;

        if (in_pad_count == -1 && out_pad_count == -1) {
            filter = filters[cur_filter_idx].first;
            separator = separators[cur_filter_idx++];
            in_pad_count = out_pad_count = 0;
        }

        if (label > separator) {
            pads.push_back(std::make_pair(in_pad_count, out_pad_count));
            in_pad_count = out_pad_count = -1;
        } else if (label < filter) {
            in_pad_count++;
            cur_label++;
        } else if (label < separator) {
            out_pad_count++;
            cur_label++;
        }
    }

    if (in_pad_count >= 0 && out_pad_count >= 0)
        pads.push_back(std::make_pair(in_pad_count, out_pad_count));

    editor_edge = 0;
    edge2pad.clear();
    filter_links.clear();
    filter_nodes.clear();

    for (unsigned i = 0; i < filters.size(); i++) {
        FilterNode node;
        std::pair <int, int> p = filters[i];
        char *opts = NULL;
        int ret;

        for (int j = p.first; j < p.second; j++) {
            if (buf.str[j] == '=') {
                opts = av_asprintf("%.*s", p.second - j-1, buf.str + j+1);
                p.second = j;
                break;
            }
        }

        filter_opts.push_back(opts);

        filter2edge.push_back(editor_edge++);
        edge2pad.push_back(Edge2Pad { i, false, false, 0, AVMEDIA_TYPE_UNKNOWN });
        for (unsigned j = 0; j < pads[i].first; j++) {
            node.inpad_edges.push_back(editor_edge);
            label2edge.push_back(editor_edge++);
            edge2pad.push_back(Edge2Pad { i, false, false, j, AVMEDIA_TYPE_UNKNOWN });
        }

        for (unsigned j = 0; j < pads[i].second; j++) {
            node.outpad_edges.push_back(editor_edge);
            label2edge.push_back(editor_edge++);
            edge2pad.push_back(Edge2Pad { i, false, true, j, AVMEDIA_TYPE_UNKNOWN });
        }

        node.id = filter_nodes.size();
        node.edge = filter2edge[i];
        node.filter_name = av_asprintf("%.*s", p.second - p.first, buf.str + p.first);
        node.filter = avfilter_get_by_name(node.filter_name);
        node.filter_label = av_asprintf("%s%d", node.filter_name, node.id);
        node.filter_options = opts;
        node.ctx_options = NULL;
        node.probe = avfilter_graph_alloc_filter(probe_graph, node.filter, "probe");
        node.ctx = NULL;
        node.pos = find_node_spot(ImVec2(300, 300));
        node.colapsed = false;
        node.set_pos = true;

        ret = av_opt_set_from_string(node.probe->priv, node.filter_options, NULL, "=", ":");
        if (ret < 0)
            av_log(NULL, AV_LOG_ERROR, "Error setting probe filter private options.\n");

        filter_nodes.push_back(node);
    }

    for (unsigned i = 0; i < labels.size(); i++) {
        std::pair <int, int> p = labels[i];

        for (unsigned j = i + 1; j < labels.size(); j++) {
            std::pair <int, int> r = labels[j];

            if ((p.second - p.first) != (r.second - r.first))
                continue;

            if (memcmp(buf.str + p.first, buf.str + r.first, p.second - p.first))
                continue;

            if (edge2pad[label2edge[i]].is_output == edge2pad[label2edge[j]].is_output)
                continue;

            filter_links.push_back(std::make_pair(label2edge[i], label2edge[j]));
        }
    }

error:
    av_bprint_finalize(&buf, NULL);

    fclose(file);

    need_filters_reinit = true;
}

static void export_filter_graph(char **out, size_t *out_size)
{
    std::vector<bool> visited;
    std::vector<unsigned> to_visit;
    AVBPrint buf;
    bool first = true;

    av_bprint_init(&buf, 512, AV_BPRINT_SIZE_UNLIMITED);

    visited.resize(filter_nodes.size());

    to_visit.push_back(0);

    while (to_visit.size() > 0) {
        unsigned node = to_visit.back();

        to_visit.pop_back();

        if (visited[node] == false) {
            visited[node] = true;

            if (first)
                first = false;
            else
                av_bprintf(&buf, ";");

            for (unsigned i = 0; i < filter_links.size(); i++) {
                const std::pair<int, int> p = filter_links[i];
                const int a = p.first;
                const int b = p.second;
                unsigned na  = edge2pad[a].node;
                bool nat = edge2pad[a].is_output;
                unsigned nb  = edge2pad[b].node;
                bool nbt = edge2pad[b].is_output;

                if (node != na && node != nb)
                    continue;
                if (node == na && nat == 1)
                    continue;
                if (node == nb && nbt == 1)
                    continue;
                av_bprintf(&buf, "[e%d]", i);
            }

            av_bprintf(&buf, "%s", filter_nodes[node].filter_name);
            if (strlen(filter_nodes[node].filter_options) > 0)
                av_bprintf(&buf, "=%s", filter_nodes[node].filter_options);

            for (unsigned i = 0; i < filter_links.size(); i++) {
                const std::pair<int, int> p = filter_links[i];
                const int a = p.first;
                const int b = p.second;
                unsigned na  = edge2pad[a].node;
                bool nat = edge2pad[a].is_output;
                unsigned nb  = edge2pad[b].node;
                bool nbt = edge2pad[b].is_output;

                if (node != na && node != nb)
                    continue;
                if (node == na && nat == 0)
                    continue;
                if (node == nb && nbt == 0)
                    continue;
                av_bprintf(&buf, "[e%d]", i);
            }

            for (unsigned i = 0; i < filter_links.size(); i++) {
                const std::pair<int, int> p = filter_links[i];
                unsigned a = edge2pad[p.first].node;
                unsigned b = edge2pad[p.second].node;

                if (node != a && node != b)
                    continue;
                if (node != a)
                    to_visit.push_back(a);
                else
                    to_visit.push_back(b);
            }
        }
    }

    av_bprintf(&buf, "\n");

    av_bprint_finalize(&buf, out);
    if (av_bprint_is_complete(&buf))
        *out_size = buf.len;
    else
        *out_size = buf.size;
    av_bprint_finalize(&buf, NULL);
}

static ImVec2 find_node_spot(ImVec2 start)
{
    ImVec2 pos = ImVec2(start.x + 80, start.y);
    bool got_pos = false;

    while (got_pos == false) {
        got_pos = true;

        for (unsigned i = 0; i < filter_nodes.size(); i++) {
            FilterNode node = filter_nodes[i];

            if (std::abs(pos.x - node.pos.x) <= 80 &&
                std::abs(pos.y - node.pos.y) <= 80) {
                got_pos = false;
                break;
            }
        }

        if (got_pos == true)
            break;

        pos.x += ((std::rand() % 3) - 1) * 100;
        pos.y += ((std::rand() % 3) - 1) * 100;
    }

    return pos;
}

static void show_filtergraph_editor(bool *p_open, bool focused)
{
    bool erased = false;
    int edge;

    if (focused)
        ImGui::SetNextWindowFocus();
    ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("FilterGraph Editor", p_open, 0)) {
        ImGui::End();
        return;
    }

    ImNodes::EditorContextSet(node_editor_context);

    ImNodes::BeginNodeEditor();

    if (ImGui::IsKeyReleased(ImGuiKey_Enter) && ImGui::GetIO().KeyCtrl)
        need_filters_reinit = true;

    const bool open_popup = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
        ImNodes::IsEditorHovered() && ((ImGui::IsKeyReleased(ImGuiKey_A) && !ImGui::GetIO().KeyShift) ||
        ImGui::IsMouseReleased(ImGuiMouseButton_Right));

    static ImVec2 click_pos;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.f, 8.f));
    if (!ImGui::IsAnyItemHovered() && open_popup) {
        ImGui::OpenPopup("Add Filter");

        click_pos = ImGui::GetMousePosOnOpeningCurrentPopup();
        click_pos.x -= ImGui::GetWindowPos().x;
        click_pos.y -= ImGui::GetWindowPos().y;
    }

    if (ImGui::BeginPopup("Add Filter")) {
        if (ImGui::BeginMenu("Source Filters", filter_graph_is_valid == false)) {
            if (ImGui::BeginMenu("Video")) {
                const AVFilter *filter = NULL;
                void *iterator = NULL;

                ImGui::SetTooltip("%s", "Video Source Filters");
                while ((filter = av_filter_iterate(&iterator))) {
                    if (!is_source_video_filter(filter))
                        continue;

                    handle_nodeitem(filter, click_pos);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Audio")) {
                const AVFilter *filter = NULL;
                void *iterator = NULL;

                ImGui::SetTooltip("%s", "Audio Source Filters");
                while ((filter = av_filter_iterate(&iterator))) {
                    if (!is_source_audio_filter(filter))
                        continue;

                    handle_nodeitem(filter, click_pos);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Media")) {
                const AVFilter *filter = NULL;
                void *iterator = NULL;

                ImGui::SetTooltip("%s", "Media Source Filters");
                while ((filter = av_filter_iterate(&iterator))) {
                    if (!is_source_media_filter(filter))
                        continue;

                    handle_nodeitem(filter, click_pos);
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Simple Filters", filter_graph_is_valid == false)) {
            if (ImGui::BeginMenu("Video")) {
                static ImGuiTextFilter imgui_filter;
                const AVFilter *filter = NULL;
                void *iterator = NULL;

                ImGui::SetTooltip("%s", "Simple Video Filters");
                imgui_filter.Draw();
                while ((filter = av_filter_iterate(&iterator))) {
                    if (!is_simple_video_filter(filter))
                        continue;

                    if (!imgui_filter.PassFilter(filter->name))
                        continue;

                    handle_nodeitem(filter, click_pos);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Audio")) {
                static ImGuiTextFilter imgui_filter;
                const AVFilter *filter = NULL;
                void *iterator = NULL;

                ImGui::SetTooltip("%s", "Simple Audio Filters");
                imgui_filter.Draw();
                while ((filter = av_filter_iterate(&iterator))) {
                    if (!is_simple_audio_filter(filter))
                        continue;

                    if (!imgui_filter.PassFilter(filter->name))
                        continue;

                    handle_nodeitem(filter, click_pos);
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Complex Filters", filter_graph_is_valid == false)) {
            static ImGuiTextFilter imgui_filter;
            const AVFilter *filter = NULL;
            void *iterator = NULL;

            imgui_filter.Draw();
            while ((filter = av_filter_iterate(&iterator))) {
                if (!is_complex_filter(filter))
                    continue;

                if (!imgui_filter.PassFilter(filter->name))
                    continue;

                handle_nodeitem(filter, click_pos);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Media Filters", filter_graph_is_valid == false)) {
            const AVFilter *filter = NULL;
            void *iterator = NULL;

            while ((filter = av_filter_iterate(&iterator))) {
                if (!is_media_filter(filter))
                    continue;

                handle_nodeitem(filter, click_pos);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Sink Filters", filter_graph_is_valid == false)) {
            if (ImGui::BeginMenu("Video")) {
                const AVFilter *filter = NULL;
                void *iterator = NULL;

                ImGui::SetTooltip("%s", "Sink Video Filters");
                while ((filter = av_filter_iterate(&iterator))) {
                    if (!is_sink_video_filter(filter))
                        continue;

                    handle_nodeitem(filter, click_pos);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Audio")) {
                const AVFilter *filter = NULL;
                void *iterator = NULL;

                ImGui::SetTooltip("%s", "Sink Audio Filters");
                while ((filter = av_filter_iterate(&iterator))) {
                    if (!is_sink_audio_filter(filter))
                        continue;

                    handle_nodeitem(filter, click_pos);
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Options")) {
            if (ImGui::BeginMenu("FilterGraph", filter_graph_is_valid == false)) {
                const char *items[] = { "quiet", "panic", "fatal", "error", "warning", "info", "verbose", "debug" };
                const int values[] = { AV_LOG_QUIET, AV_LOG_PANIC, AV_LOG_FATAL, AV_LOG_ERROR, AV_LOG_WARNING, AV_LOG_INFO, AV_LOG_VERBOSE, AV_LOG_DEBUG };
                static int item_current_idx = 5;

                ImGui::InputInt("Max Number of FilterGraph Threads", &filter_graph_nb_threads);
                ImGui::InputInt("Auto Conversion Type for FilterGraph", &filter_graph_auto_convert_flags);
                if (ImGui::BeginCombo("Log Message Level", items[item_current_idx], 0)) {
                    for (int n = 0; n < IM_ARRAYSIZE(items); n++) {
                        const bool is_selected = (item_current_idx == n);

                        if (ImGui::Selectable(items[n], is_selected)) {
                            item_current_idx = n;
                            log_level = values[n];
                        }

                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Video Outputs")) {
                const char *items[] = { "nearest", "linear" };
                const GLint values[] = { GL_NEAREST, GL_LINEAR };
                static int item_current_idx[2] = { 0, 0 };
                const int flags = 0;

                if (ImGui::BeginCombo("Upscaler", items[item_current_idx[0]], flags)) {
                    for (int n = 0; n < IM_ARRAYSIZE(items); n++) {
                        const bool is_selected = (item_current_idx[0] == n);

                        if (ImGui::Selectable(items[n], is_selected)) {
                            item_current_idx[0] = n;
                            global_upscale_interpolation = values[n];
                        }

                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                if (ImGui::BeginCombo("Downscaler", items[item_current_idx[1]], flags)) {
                    for (int n = 0; n < IM_ARRAYSIZE(items); n++) {
                        const bool is_selected = (item_current_idx[1] == n);

                        if (ImGui::Selectable(items[n], is_selected)) {
                            item_current_idx[1] = n;
                            global_downscale_interpolation = values[n];
                        }

                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Audio Outputs")) {
                ImGui::DragFloat2("Sample Range", audio_sample_range, 0.01f, 1.f, 8.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                ImGui::InputFloat2("Window Size", audio_window_size);
                if (ImGui::DragFloat3("Listener Position", listener_position, 0.01f, -1.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput))
                    alListenerfv(AL_POSITION, listener_position);
                if (ImGui::DragFloat3("Listener Direction At", listener_direction, 0.01f, -1.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput))
                    alListenerfv(AL_ORIENTATION, listener_direction);
                if (ImGui::DragFloat3("Listener Direction Up", &listener_direction[3], 0.01f, -1.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput))
                    alListenerfv(AL_ORIENTATION, listener_direction);

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("MiniMap")) {
                const char *items[] = { "Off", "On" };
                const bool values[] = { false, true };
                const int positions[] = {
                    ImNodesMiniMapLocation_BottomLeft,
                    ImNodesMiniMapLocation_BottomRight,
                    ImNodesMiniMapLocation_TopLeft,
                    ImNodesMiniMapLocation_TopRight,
                };
                const char *positions_name[] = {
                    "Bottom Left",
                    "Bottom Right",
                    "Top Left",
                    "Top Right",
                };
                const int flags = 0;

                if (ImGui::BeginCombo("Toggle Display", items[show_mini_map], flags)) {
                    for (int n = 0; n < IM_ARRAYSIZE(items); n++) {
                        const bool is_selected = values[n] == show_mini_map;

                        if (ImGui::Selectable(items[n], is_selected))
                            show_mini_map = values[n];

                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                if (ImGui::BeginCombo("Corner Position", positions_name[mini_map_location], flags)) {
                    for (int n = 0; n < IM_ARRAYSIZE(positions); n++) {
                        const bool is_selected = mini_map_location == positions[n];

                        if (ImGui::Selectable(positions_name[n], is_selected))
                            mini_map_location = positions[n];

                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Display", filter_graph_is_valid == false)) {
                const char *items[] = { "Windowed", "Fullscreen" };
                const bool values[] = { false, true };
                const bool old_display = full_screen;

                if (ImGui::BeginCombo("Screen mode", items[full_screen], 0)) {
                    for (int n = 0; n < IM_ARRAYSIZE(values); n++) {
                        const bool is_selected = full_screen == values[n];

                        if (ImGui::Selectable(items[n], is_selected))
                            full_screen = values[n];

                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                if (old_display != full_screen)
                    restart_display = true;
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Visual Color Style")) {
                if (ImGui::MenuItem("Classic")) {
                    ImGui::StyleColorsClassic();
                    ImNodes::StyleColorsClassic();
                }
                if (ImGui::MenuItem("Dark")) {
                    ImGui::StyleColorsDark();
                    ImNodes::StyleColorsDark();
                }
                if (ImGui::MenuItem("Light")) {
                    ImGui::StyleColorsLight();
                    ImNodes::StyleColorsLight();
                }
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Export FilterGraph", filter_graph_is_valid == true)) {
            if (ImGui::BeginMenu("Save as Script")) {
                static char file_name[1024] = { 0 };
                size_t out_size = 0;
                char *out = NULL;

                ImGui::InputText("File name:", file_name, sizeof(file_name) - 1);
                if (strlen(file_name) > 0 && ImGui::Button("Save")) {
                    export_filter_graph(&out, &out_size);

                    if (out && out_size > 0) {
                        FILE *script_file = fopen(file_name, "w");

                        if (script_file) {
                            fwrite(out, 1, out_size, script_file);
                            fclose(script_file);
                        }
                        av_freep(&out);
                        out_size = 0;
                        memset(file_name, 0, sizeof(file_name));
                    }
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Import FilterGraph", filter_graph_is_valid == false &&
                             filter_links.size() == 0 &&
                             filter_nodes.size() == 0)) {
            if (ImGui::BeginMenu("Load Script")) {
                static char file_name[1024] = { 0 };

                ImGui::InputText("File name:", file_name, sizeof(file_name) - 1);
                if (strlen(file_name) > 0 && ImGui::Button("Load")) {
                    av_freep(&import_script_file_name);
                    import_script_file_name = av_asprintf("%s", file_name);
                    memset(file_name, 0, sizeof(file_name));
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }

        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();

    edge2pad.resize(editor_edge);

    for (unsigned i = 0; i < filter_nodes.size(); i++) {
        FilterNode *filter_node = &filter_nodes[i];

        edge = filter_node->edge;
        edge2pad[edge] = (Edge2Pad { i, false, false, 0, AVMEDIA_TYPE_UNKNOWN });
        ImNodes::BeginNode(filter_node->edge);
        if (filter_node->set_pos) {
            ImNodes::SetNodeEditorSpacePos(filter_node->edge, filter_node->pos);
            filter_node->set_pos = false;
        } else {
            filter_node->pos = ImNodes::GetNodeEditorSpacePos(filter_node->edge);
        }
        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(filter_node->filter_name);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s\n%s", filter_node->filter_label, filter_node->filter->description);
        ImNodes::EndNodeTitleBar();
        draw_node_options(filter_node);
        if (!filter_node->probe) {
            ImNodes::EndNode();
            continue;
        }

        AVFilterContext *filter_ctx = filter_node->ctx ? filter_node->ctx : filter_node->probe;

        for (unsigned j = 0; j < filter_node->inpad_edges.size(); j++) {
            const int edge = filter_node->inpad_edges[j];

            edge2pad[edge].removed = true;
        }
        filter_node->inpad_edges.resize(static_cast<size_t>(filter_ctx->nb_inputs));

        for (unsigned j = 0; j < filter_node->outpad_edges.size(); j++) {
            const int edge = filter_node->outpad_edges[j];

            edge2pad[edge].removed = true;
        }
        filter_node->outpad_edges.resize(static_cast<size_t>(filter_ctx->nb_outputs));

        for (unsigned j = 0; j < filter_ctx->nb_inputs; j++) {
            enum AVMediaType media_type;

            edge = filter_node->inpad_edges[j];
            if (edge == 0) {
                edge = editor_edge++;
                filter_node->inpad_edges[j] = edge;
                edge2pad.resize(editor_edge);
            }
            media_type = avfilter_pad_get_type(filter_ctx->input_pads, j);
            if (media_type == AVMEDIA_TYPE_VIDEO) {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(  0, 255, 255, 255));
            } else {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(255, 255,   0, 255));
            }
            edge2pad[edge] = (Edge2Pad { i, false, false, j, media_type });
            filter_node->inpad_edges[j] = edge;
            ImNodes::BeginInputAttribute(edge);
            ImGui::Text("%s", avfilter_pad_get_name(filter_ctx->input_pads, j));
            ImNodes::EndInputAttribute();
            ImNodes::PopColorStyle();
        }

        for (unsigned j = 0; j < filter_ctx->nb_outputs; j++) {
            enum AVMediaType media_type;

            edge = filter_node->outpad_edges[j];
            if (edge == 0) {
                edge = editor_edge++;
                filter_node->outpad_edges[j] = edge;
                edge2pad.resize(editor_edge);
            }
            media_type = avfilter_pad_get_type(filter_ctx->output_pads, j);
            if (media_type == AVMEDIA_TYPE_VIDEO) {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(  0, 255, 255, 255));
            } else {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(255, 255,   0, 255));
            }
            edge2pad[edge] = (Edge2Pad { i, false, true, j, media_type });
            filter_node->outpad_edges[j] = edge;
            ImNodes::BeginOutputAttribute(edge);
            ImGui::Text("%s", avfilter_pad_get_name(filter_ctx->output_pads, j));
            ImNodes::EndOutputAttribute();
            ImNodes::PopColorStyle();
        }

        ImNodes::EndNode();
        ImNodes::SetNodeDraggable(filter_node->edge, true);
    }

    for (unsigned i = 0; i < filter_links.size(); i++) {
        const std::pair<int, int> p = filter_links[i];

        if (edge2pad[p.first].removed  == true ||
            edge2pad[p.second].removed == true) {
            edge2pad[p.first].removed    = true;
            edge2pad[p.second].removed   = true;
            edge2pad[p.first].is_output  = false;
            edge2pad[p.second].is_output = false;

            filter_links.erase(filter_links.begin() + i);
            continue;
        }

        if (edge2pad[p.first].type  == AVMEDIA_TYPE_UNKNOWN ||
            edge2pad[p.second].type == AVMEDIA_TYPE_UNKNOWN) {
            edge2pad[p.first].removed    = true;
            edge2pad[p.second].removed   = true;
            edge2pad[p.first].is_output  = false;
            edge2pad[p.second].is_output = false;

            filter_links.erase(filter_links.begin() + i);
            continue;
        }

        if (edge2pad[p.first].is_output == edge2pad[p.second].is_output) {
            edge2pad[p.first].removed    = true;
            edge2pad[p.second].removed   = true;
            edge2pad[p.first].is_output  = false;
            edge2pad[p.second].is_output = false;

            filter_links.erase(filter_links.begin() + i);
            continue;
        }

        ImNodes::Link(i, p.first, p.second);
    }

    if (show_mini_map == true)
        ImNodes::MiniMap(0.2f, mini_map_location);
    ImNodes::EndNodeEditor();

    for (unsigned i = 0; i < video_sink_threads.size(); i++) {
        if (video_sink_threads[i].joinable()) {
            ImGui::End();
            return;
        }
    }

    for (unsigned i = 0; i < audio_sink_threads.size(); i++) {
        if (audio_sink_threads[i].joinable()) {
            ImGui::End();
            return;
        }
    }

    int link_id;
    if (ImNodes::IsLinkDestroyed(&link_id) && filter_links.size() > 0) {
        const std::pair<int, int> p = filter_links[link_id];

        edge2pad[p.first].removed    = true;
        edge2pad[p.second].removed   = true;
        edge2pad[p.first].is_output  = false;
        edge2pad[p.second].is_output = false;

        filter_links.erase(filter_links.begin() + link_id);
    }

    const int links_selected = ImNodes::NumSelectedLinks();
    if (!ImGui::IsItemHovered() && links_selected > 0 && ImGui::IsKeyReleased(ImGuiKey_X) && filter_links.size() > 0) {
        std::vector<int> selected_links;

        selected_links.resize(static_cast<size_t>(links_selected));
        ImNodes::GetSelectedLinks(selected_links.data());
        std::sort(selected_links.begin(), selected_links.end(), std::greater <>());

        for (const int link_id : selected_links) {
            const std::pair<int, int> p = filter_links[link_id];

            edge2pad[p.first].removed    = true;
            edge2pad[p.second].removed   = true;
            edge2pad[p.first].is_output  = false;
            edge2pad[p.second].is_output = false;

            filter_links.erase(filter_links.begin() + link_id);
        }
    }

    const unsigned nodes_selected = ImNodes::NumSelectedNodes();
    if (nodes_selected > 0 && nodes_selected <= filter_nodes.size() && !ImGui::IsItemHovered() && ImGui::IsKeyReleased(ImGuiKey_X) && ImGui::GetIO().KeyShift) {
        std::vector<int> selected_nodes;

        selected_nodes.resize(static_cast<size_t>(nodes_selected));
        ImNodes::GetSelectedNodes(selected_nodes.data());

        for (unsigned node_id = 0; node_id < selected_nodes.size(); node_id++) {
            const int edge = selected_nodes[node_id];
            if (edge < 0)
                continue;
            const unsigned node = edge2pad[edge].node;
            std::vector<int> removed_edges;

            edge2pad[edge].removed = true;
            edge2pad[edge].is_output = false;
            filter_nodes[node].filter = NULL;
            avfilter_free(filter_nodes[node].ctx);
            filter_nodes[node].ctx = NULL;
            av_freep(&filter_nodes[node].filter_name);
            av_freep(&filter_nodes[node].filter_label);
            av_freep(&filter_nodes[node].filter_options);
            av_freep(&filter_nodes[node].ctx_options);
            avfilter_free(filter_nodes[node].probe);
            filter_nodes[node].probe = NULL;

            removed_edges.push_back(filter_nodes[node].edge);
            for (unsigned j = 0; j < filter_nodes[node].inpad_edges.size(); j++)
                removed_edges.push_back(filter_nodes[node].inpad_edges[j]);

            for (unsigned j = 0; j < filter_nodes[node].outpad_edges.size(); j++)
                removed_edges.push_back(filter_nodes[node].outpad_edges[j]);

            erased = true;

            for (unsigned r = 0; r < removed_edges.size(); r++) {
                std::vector<unsigned> selected_links;
                const int removed_edge = removed_edges[r];

                if (removed_edge < 0)
                    continue;
                edge2pad[removed_edge].type = AVMEDIA_TYPE_UNKNOWN;
                edge2pad[removed_edge].is_output = false;
                edge2pad[removed_edge].removed = true;

                if (filter_links.size() <= 0)
                    continue;

                for (unsigned l = 0; l < filter_links.size(); l++) {
                    const std::pair<int, int> p = filter_links[l];

                    if (p.first == removed_edge || p.second == removed_edge)
                        selected_links.push_back(l);
                }

                std::sort(selected_links.begin(), selected_links.end(), std::greater <>());
                for (const unsigned link_id : selected_links) {
                    const std::pair<int, int> p = filter_links[link_id];

                    edge2pad[p.first].removed    = true;
                    edge2pad[p.second].removed   = true;
                    edge2pad[p.first].is_output  = false;
                    edge2pad[p.second].is_output = false;

                    filter_links.erase(filter_links.begin() + link_id);
                }
            }
        }
    }

    if (erased && filter_nodes.size() > 0) {
        unsigned i = filter_nodes.size() - 1;
        do {
            if (!filter_nodes[i].filter)
                filter_nodes.erase(filter_nodes.begin() + i);
            else
                filter_nodes[i].set_pos = true;
        } while (i--);
    }

    const unsigned copy_nodes_selected = ImNodes::NumSelectedNodes();
    if (copy_nodes_selected > 0 && copy_nodes_selected <= filter_nodes.size() && !ImGui::IsItemHovered() && ImGui::IsKeyReleased(ImGuiKey_C) && ImGui::GetIO().KeyShift) {
        std::vector<int> selected_nodes;

        selected_nodes.resize(static_cast<size_t>(copy_nodes_selected));
        ImNodes::GetSelectedNodes(selected_nodes.data());

        for (unsigned i = 0; i < selected_nodes.size(); i++) {
            const int e = selected_nodes[i];
            if (e < 0)
                continue;
            FilterNode orig = filter_nodes[edge2pad[e].node];
            FilterNode copy;

            copy.filter = orig.filter;
            copy.id = filter_nodes.size();
            copy.filter_name = av_strdup(orig.filter->name);
            copy.filter_label = av_asprintf("%s%d", copy.filter->name, copy.id);
            copy.filter_options = NULL;
            copy.ctx_options = NULL;
            copy.probe = avfilter_graph_alloc_filter(probe_graph, copy.filter, "probe");
            copy.ctx = NULL;
            copy.pos = find_node_spot(orig.pos);
            copy.colapsed = false;
            copy.set_pos = true;
            copy.edge = editor_edge++;

            av_opt_copy(copy.probe, orig.probe);
            av_opt_copy(copy.probe->priv, orig.probe->priv);

            edge2pad.push_back(Edge2Pad { copy.id, false, false, 0, AVMEDIA_TYPE_UNKNOWN });

            for (unsigned j = 0; j < orig.inpad_edges.size(); j++) {
                copy.inpad_edges.push_back(editor_edge++);
                edge2pad.push_back(Edge2Pad { copy.id, false, false, j, AVMEDIA_TYPE_UNKNOWN });
            }

            for (unsigned j = 0; j < orig.outpad_edges.size(); j++) {
                copy.outpad_edges.push_back(editor_edge++);
                edge2pad.push_back(Edge2Pad { copy.id, false, true, j, AVMEDIA_TYPE_UNKNOWN });
            }

            filter_nodes.push_back(copy);
        }
    }

    if (!ImGui::IsItemHovered() && ImGui::IsKeyReleased(ImGuiKey_A) && ImGui::GetIO().KeyShift) {
        const AVFilter *buffersink  = avfilter_get_by_name("buffersink");
        const AVFilter *abuffersink = avfilter_get_by_name("abuffersink");
        std::vector<int> unconnected_edges;
        std::vector<int> connected_edges;

        for (unsigned l = 0; l < filter_links.size(); l++) {
            const std::pair<int, int> p = filter_links[l];

            connected_edges.push_back(p.first);
            connected_edges.push_back(p.second);
        }

        for (int e = 0; e < editor_edge; e++) {
            unsigned ee = 0;
            for (; ee < connected_edges.size(); ee++) {
                if (e == connected_edges[ee])
                    break;
            }

            if (ee == connected_edges.size() &&
                edge2pad[e].type != AVMEDIA_TYPE_UNKNOWN &&
                edge2pad[e].is_output == true &&
                edge2pad[e].removed == false) {
                unconnected_edges.push_back(e);
            }
        }

        for (unsigned i = 0; i < unconnected_edges.size(); i++) {
            const int e = unconnected_edges[i];
            if (e < 0)
                continue;
            enum AVMediaType type = edge2pad[e].type;
            FilterNode src = filter_nodes[edge2pad[e].node];
            FilterNode node;

            node.filter = type == AVMEDIA_TYPE_AUDIO ? abuffersink : buffersink;
            node.id = filter_nodes.size();
            node.filter_name = av_strdup(node.filter->name);
            node.filter_label = av_asprintf("%s%d", node.filter->name, node.id);
            node.filter_options = NULL;
            node.ctx_options = NULL;
            node.probe = NULL;
            node.ctx = NULL;
            node.pos = find_node_spot(src.pos);
            node.colapsed = false;
            node.set_pos = true;
            node.edge = editor_edge++;
            node.inpad_edges.push_back(editor_edge);
            edge2pad.push_back(Edge2Pad { node.id, false, false, 0, AVMEDIA_TYPE_UNKNOWN });
            edge2pad.push_back(Edge2Pad { node.id, false, false, 0, AVMEDIA_TYPE_UNKNOWN });

            filter_nodes.push_back(node);
            filter_links.push_back(std::make_pair(e, editor_edge++));
        }
    }

    int start_attr, end_attr;
    if (ImNodes::IsLinkCreated(&start_attr, &end_attr)) {
        const enum AVMediaType first  = edge2pad[start_attr].type;
        const enum AVMediaType second = edge2pad[end_attr].type;

        if (first == second && first != AVMEDIA_TYPE_UNKNOWN)
            filter_links.push_back(std::make_pair(start_attr, end_attr));
    }

    ImGui::End();
}

static void draw_filters_commands(unsigned *toggle_filter)
{
    static unsigned selected_filter = -1;

    if (ImGui::BeginListBox("##Commands Box", ImVec2(-1, -1))) {
        static ImGuiTextFilter imgui_filter;

        imgui_filter.Draw();
        for (unsigned n = 0; n < filter_nodes.size(); n++) {
            const AVFilterContext *ctx = filter_nodes[n].ctx;
            const bool is_selected = selected_filter == n;
            static bool is_opened = false;
            static bool clean_storage = true;

            if (!ctx)
                continue;

            if (!ctx->filter)
                continue;

            if (!imgui_filter.PassFilter(ctx->filter->name))
                continue;

            if (ImGui::Selectable(ctx->filter->name, is_selected)) {
                selected_filter = n;
            }

            if (ImGui::IsItemActive() || ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", ctx->name);
            }

            if (ImGui::IsItemClicked() && ImGui::IsItemActive()) {
                selected_filter = n;
                is_opened = true;
            }

            if (is_opened && selected_filter == n)
                draw_filter_commands(ctx, n, toggle_filter, is_opened, &clean_storage, true);
        }

        ImGui::EndListBox();
    }
}

static void show_commands(bool *p_open, bool focused)
{
    static unsigned toggle_filter = UINT_MAX;

    if (filter_graph_is_valid == false || (
        ((buffer_sinks.size() != mutexes.size() ||
          buffer_sinks.size() == 0)) &&
        ((abuffer_sinks.size() != amutexes.size() ||
          abuffer_sinks.size() == 0))))
        return;

    if (focused)
        ImGui::SetNextWindowFocus();
    if (!ImGui::Begin("Filter Commands", p_open, 0)) {
        ImGui::End();
        return;
    }

    draw_filters_commands(&toggle_filter);

    ImGui::End();

    if (toggle_filter < UINT_MAX) {
        const AVFilterContext *filter_ctx = filter_graph->filters[toggle_filter];
        const int flag = !filter_ctx->is_disabled;

        avfilter_graph_send_command(filter_graph, filter_ctx->name, "enable", flag ? "0" : "1", NULL, 0, 0);
        toggle_filter = UINT_MAX;
    }
}

static void show_dumpgraph(bool *p_open, bool focused)
{
    if (!graphdump_text || filter_graph_is_valid == false)
        return;

    if (focused)
        ImGui::SetNextWindowFocus();
    if (!ImGui::Begin("FilterGraph Dump", p_open, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        return;
    }
    ImGui::Text("%s", graphdump_text);
    ImGui::End();
}

static void log_callback(void *ptr, int level, const char *fmt, va_list args)
{
    log_mutex.lock();

    if (log_level >= level) {
        int old_size = log_buffer.size();

        switch (level) {
        case AV_LOG_QUIET:
            break;
        case AV_LOG_PANIC:
            log_buffer.appendf("[panic] ");
            break;
        case AV_LOG_FATAL:
            log_buffer.appendf("[fatal] ");
            break;
        case AV_LOG_ERROR:
            log_buffer.appendf("[error] ");
            break;
        case AV_LOG_WARNING:
            log_buffer.appendf("[warning] ");
            break;
        case AV_LOG_INFO:
            log_buffer.appendf("[info] ");
            break;
        case AV_LOG_VERBOSE:
            log_buffer.appendf("[verbose] ");
            break;
        case AV_LOG_DEBUG:
            log_buffer.appendf("[debug] ");
            break;
        default:
            log_buffer.appendf("[unknown%d] ", level);
            break;
        }
        log_buffer.appendfv(fmt, args);
        for (int new_size = log_buffer.size(); old_size < new_size; old_size++)
            if (log_buffer[old_size] == '\n')
                log_lines_offsets.push_back(old_size + 1);
    }

    log_mutex.unlock();
}

static void show_log(bool *p_open, bool focused)
{
    static ImGuiTextFilter filter;

    if (focused)
        ImGui::SetNextWindowFocus();
    ImGui::SetNextWindowSize(ImVec2(500, 100), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("FilterGraph Log", p_open, 0)) {
        ImGui::End();
        return;
    }

    filter.Draw("###Log Filter", ImGui::GetWindowSize().x);
    if (filter.IsActive()) {
        for (int line_no = 0; line_no < log_lines_offsets.Size; line_no++) {
            const char *line_start = log_buffer.begin() + log_lines_offsets[line_no];
            const char *line_end = (line_no + 1 < log_lines_offsets.Size) ? (log_buffer.begin() + log_lines_offsets[line_no + 1] - 1) : log_buffer.end();
            if (filter.PassFilter(line_start, line_end))
                ImGui::TextUnformatted(line_start, line_end);
        }
    } else {
        ImGui::TextUnformatted(log_buffer.begin(), log_buffer.end());
    }

    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);

    ImGui::End();
}

int main(int, char**)
{
    ALCint attribs[] = { ALC_FREQUENCY, output_sample_rate, 0, 0 };

    al_dev = alcOpenDevice(NULL);
    if (!al_dev) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open AL device.\n");
        return -1;
    }

    al_ctx = alcCreateContext(al_dev, attribs);
    alcMakeContextCurrent(al_ctx);
    alListenerfv(AL_POSITION, listener_position);
    alListenerfv(AL_ORIENTATION, listener_direction);

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return -1;

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

restart_window:
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    if (monitor) {
        const GLFWvidmode *mode = glfwGetVideoMode(monitor);

        width = mode->width;
        height = mode->height;
    }

    GLFWwindow *window = glfwCreateWindow(width, height, "lavfi-preview", full_screen ? monitor : NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    av_log_set_callback(log_callback);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImNodes::CreateContext();

    node_editor_context = ImNodes::EditorContextCreate();
    if (node_editor_context == NULL)
        return 1;

    ImGuiIO& io = ImGui::GetIO(); (void)io;

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.WantCaptureKeyboard = true;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != NULL);
    // Our state

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        int64_t min_aqpts = INT64_MAX;
        int64_t min_qpts = INT64_MAX;

        glfwPollEvents();

        if (show_abuffersink_window == false) {
            if (audio_sink_threads.size() > 0) {
                need_filters_reinit = true;

                if (play_sound_thread.joinable())
                    play_sound_thread.join();
                play_sources.clear();

                kill_audio_sink_threads();

                audio_sink_threads.clear();

                need_filters_reinit = false;
            }
        }

        if (show_buffersink_window == false) {
            if (video_sink_threads.size() > 0) {
                need_filters_reinit = true;

                kill_video_sink_threads();

                video_sink_threads.clear();

                need_filters_reinit = false;
            }
        }

        if (import_script_file_name != NULL) {
            import_filter_graph(import_script_file_name);
            av_freep(&import_script_file_name);
        }

        filters_setup();

        if (filter_graph_is_valid) {
            for (unsigned i = 0; i < buffer_sinks.size(); i++) {
                BufferSink *sink = &buffer_sinks[i];

                if (sink->pts == AV_NOPTS_VALUE) {
                    min_qpts = sink->qpts = AV_NOPTS_VALUE;
                    continue;
                }
                sink->qpts = av_rescale_q(sink->pts, sink->time_base, AV_TIME_BASE_Q);
                min_qpts = std::min(min_qpts, sink->qpts);
            }
        }

        if (filter_graph_is_valid) {
            for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
                BufferSink *sink = &abuffer_sinks[i];

                if (sink->pts == AV_NOPTS_VALUE) {
                    min_aqpts = sink->qpts = AV_NOPTS_VALUE;
                    continue;
                }
                sink->qpts = av_rescale_q(sink->pts, sink->time_base, AV_TIME_BASE_Q);
                min_aqpts = std::min(min_aqpts, sink->qpts);
            }
        }

        min_qpts = std::min(min_qpts, min_aqpts);
        min_aqpts = min_qpts;

        if (filter_graph_is_valid) {
            for (unsigned i = 0; i < buffer_sinks.size(); i++) {
                BufferSink *sink = &buffer_sinks[i];
                AVFrame *render_frame = NULL;

                if (ring_buffer_num_items(&sink->render_frames) > sink->render_ring_size - 1)
                    continue;

                if (sink->qpts > min_qpts)
                    continue;

                { std::lock_guard lk(mutexes[i]); sink->ready = true; }
                cv[i].notify_one();
                ring_buffer_dequeue(&sink->consume_frames, &render_frame);
                if (!render_frame)
                    continue;
                ring_buffer_enqueue(&sink->render_frames, render_frame);
            }
        }

        if (filter_graph_is_valid) {
            for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
                BufferSink *sink = &abuffer_sinks[i];
                AVFrame *render_frame = NULL;

                if (sink->unprocessed_bufids.size() == 0) {
                    ALint queued = 0;

                    alGetSourcei(sink->source, AL_BUFFERS_QUEUED, &queued);
                    if (queued > AL_BUFFERS / 2)
                        continue;

                    if (queued < AL_BUFFERS / 2)
                        goto dequeue_consume_frames;

                    if (ring_buffer_num_items(&sink->render_frames) > sink->render_ring_size - 1)
                        continue;

                    if (sink->qpts > min_aqpts)
                        continue;
                }

dequeue_consume_frames:
                { std::lock_guard lk(amutexes[i]); sink->ready = true; }
                acv[i].notify_one();
                ring_buffer_dequeue(&sink->consume_frames, &render_frame);
                if (!render_frame)
                    continue;
                ring_buffer_enqueue(&sink->render_frames, render_frame);
            }
        }

        if (filter_graph_is_valid && show_abuffersink_window == true) {
            for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
                BufferSink *sink = &abuffer_sinks[i];
                AVFrame *play_frame = NULL;

                ring_buffer_peek(&sink->render_frames, &play_frame, 0);
                if (play_frame) {
                    const float *src = (const float *)play_frame->extended_data[0];

                    if (src && play_frame->nb_samples > 0) {
                        float min = FLT_MAX, max = -FLT_MAX;

                        for (int n = 0; n < play_frame->nb_samples; n++) {
                            max = std::max(max, src[n]);
                            min = std::min(min, src[n]);
                        }

                        sink->frame_number++;
                        sink->pts = play_frame->pts;
                        sink->pos = play_frame->pkt_pos;
                        sink->frame_nb_samples = play_frame->nb_samples;
                        sink->samples[sink->sample_index++] = max;
                        sink->samples[sink->sample_index++] = min;
                        if (sink->sample_index >= sink->nb_samples)
                            sink->sample_index = 0;
                    }

                    queue_sound(play_frame, sink);
                }
            }
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (filter_graph_is_valid && show_buffersink_window == true) {
            for (unsigned i = 0; i < buffer_sinks.size(); i++) {
                BufferSink *sink = &buffer_sinks[i];
                AVFrame *render_frame = NULL;

                ring_buffer_peek(&sink->render_frames, &render_frame, 0);
                if (!render_frame)
                    continue;

                draw_frame(&sink->texture, &show_buffersink_window, render_frame, sink);
            }
        }

        if (filter_graph_is_valid && show_abuffersink_window == true) {
            for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
                BufferSink *sink = &abuffer_sinks[i];

                draw_aframe(&show_abuffersink_window, sink);
            }
        }

        bool focused = ImGui::IsKeyReleased(ImGuiKey_F3);
        if (focused)
            show_commands_window = true;
        if (show_commands_window)
            show_commands(&show_commands_window, focused);
        focused = ImGui::IsKeyReleased(ImGuiKey_F4);
        if (focused)
            show_dumpgraph_window = true;
        if (show_dumpgraph_window)
            show_dumpgraph(&show_dumpgraph_window, focused);
        focused = ImGui::IsKeyReleased(ImGuiKey_F5);
        if (focused)
            show_log_window = true;
        if (show_log_window)
            show_log(&show_log_window, focused);
        focused = ImGui::IsKeyReleased(ImGuiKey_F2);
        if (focused)
            show_filtergraph_editor_window = true;
        if (show_filtergraph_editor_window)
            show_filtergraph_editor(&show_filtergraph_editor_window, focused);
        show_help = ImGui::IsKeyDown(ImGuiKey_F1);
        if (show_help)
            draw_help(&show_help);
        show_info = ImGui::IsKeyDown(ImGuiKey_I) && !io.WantTextInput;
        if (show_info)
            draw_info(&show_info, &frame_info);
        show_console ^= ImGui::IsKeyReleased(ImGuiKey_Escape);
        if (show_console)
            draw_console(&show_console);

        // Rendering
        ImGui::Render();
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        if (filter_graph_is_valid) {
            for (unsigned i = 0; i < buffer_sinks.size(); i++) {
                BufferSink *sink = &buffer_sinks[i];
                AVFrame *purge_frame = NULL;

                if (paused && !framestep)
                    continue;

                if (sink->qpts > min_qpts)
                    continue;

                if (ring_buffer_num_items(&sink->render_frames) < sink->render_ring_size)
                    continue;

                ring_buffer_dequeue(&sink->render_frames, &purge_frame);
                if (!purge_frame)
                    continue;
                ring_buffer_enqueue(&sink->purge_frames, purge_frame);

                clear_ring_buffer(&sink->purge_frames);
            }
        }

        if (filter_graph_is_valid) {
            for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
                BufferSink *sink = &abuffer_sinks[i];
                AVFrame *purge_frame = NULL;

                if (sink->unprocessed_bufids.size() == 0) {
                    ALint queued = 0;

                    alGetSourcei(sink->source, AL_BUFFERS_QUEUED, &queued);
                    if (queued > AL_BUFFERS / 2)
                        continue;

                    if (paused && !framestep)
                        continue;

                    if (queued < AL_BUFFERS / 2)
                        goto dequeue_render_frames;

                    if (sink->qpts > min_aqpts)
                        continue;

                    if (ring_buffer_num_items(&sink->render_frames) < sink->render_ring_size)
                        continue;
                }

dequeue_render_frames:
                ring_buffer_dequeue(&sink->render_frames, &purge_frame);
                if (!purge_frame)
                    continue;
                ring_buffer_enqueue(&sink->purge_frames, purge_frame);

                clear_ring_buffer(&sink->purge_frames);
            }
        }

        if (restart_display == true)
            break;
    }

    need_filters_reinit = true;

    if (play_sound_thread.joinable())
        play_sound_thread.join();
    play_sources.clear();

    kill_audio_sink_threads();
    kill_video_sink_threads();

    video_sink_threads.clear();
    audio_sink_threads.clear();

    for (unsigned i = 0; i < filter_nodes.size(); i++) {
        FilterNode *node = &filter_nodes[i];

        av_freep(&node->filter_name);
        av_freep(&node->filter_label);
        av_freep(&node->filter_options);
        av_freep(&node->ctx_options);
        avfilter_free(node->probe);
        node->probe = NULL;
        node->ctx = NULL;
    }

    filter_nodes.clear();

    av_freep(&graphdump_text);

    avfilter_graph_free(&filter_graph);
    avfilter_graph_free(&probe_graph);

    filter_links.clear();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImNodes::EditorContextFree(node_editor_context);
    node_editor_context = NULL;
    ImNodes::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    if (restart_display == true) {
        restart_display = false;
        goto restart_window;
    }

    glfwTerminate();

    alcDestroyContext(al_ctx);
    al_ctx = NULL;
    alcCloseDevice(al_dev);
    al_dev = NULL;

    return 0;
}
