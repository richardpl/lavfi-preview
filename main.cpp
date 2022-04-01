#include <thread>
#include <mutex>
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
#include <libavutil/time.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavformat/avformat.h>
#include "ringbuffer/ringbuffer.c"
}

#define AL_BUFFERS 64

typedef struct Edge2Pad {
    unsigned node;
    bool is_output;
    unsigned pad_index;
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
    bool fullscreen;
    bool show_osd;
    bool need_more;
    bool have_window_pos;
    ImVec2 window_pos;
    GLuint texture;
    ALuint source;
    ALenum format;
    ALuint buffers[AL_BUFFERS];
    std::vector<ALuint> processed_bufids;
    std::vector<ALuint> unprocessed_bufids;

    GLint downscale_interpolator;
    GLint upscale_interpolator;

    int64_t pts;
    float *samples;
    unsigned nb_samples;
    unsigned sample_index;
} BufferSink;

typedef struct FilterNode {
    int id;
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
    AVFilterGraph   *probe_graph;
    AVFilterContext *ctx;

    std::vector<OptStorage> opt_storage;
} FilterNode;

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int filter_graph_nb_threads = 0;
int filter_graph_auto_convert_flags = 0;
unsigned focus_buffersink_window = -1;
unsigned focus_abuffersink_window = -1;
bool show_abuffersink_window = true;
bool show_buffersink_window = true;
bool show_dumpgraph_window = true;
bool show_commands_window = true;
bool show_filtergraph_editor_window = true;
bool show_mini_map = true;
int mini_map_location = ImNodesMiniMapLocation_BottomRight;

bool need_filters_reinit = true;
bool framestep = false;
bool paused = true;
bool show_help = false;

GLint global_upscale_interpolation = GL_NEAREST;
GLint global_downscale_interpolation = GL_NEAREST;

int output_sample_rate = 44100;
int display_w;
int display_h;
int width = 1280;
int height = 720;
bool filter_graph_is_valid = false;
AVFilterGraph *filter_graph = NULL;
char *graphdump_text = NULL;

ImNodesEditorContext *node_editor_context;

std::mutex filtergraph_mutex;

std::vector<BufferSink> abuffer_sinks;
std::vector<BufferSink> buffer_sinks;
std::vector<std::mutex> amutexes;
std::vector<std::mutex> mutexes;
std::vector<std::thread> audio_sink_threads;
std::vector<std::thread> video_sink_threads;
std::vector<FilterNode> filter_nodes;
std::vector<std::pair<int, int>> filter_links;
std::vector<std::pair<int, enum AVMediaType>> edge2type;
std::vector<Edge2Pad> edge2pad;

static const enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_RGBA, AV_PIX_FMT_NONE };
static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_NONE };
static const int sample_rates[] = { 44100, 0 };

ALCdevice *al_dev = NULL;
ALCcontext *al_ctx = NULL;
float direction[6] = { 0, 0, -1, 0, 1, 0 };
float position[3] = { 0, 0, 0 };

static void clear_ring_buffer(ring_buffer_t *ring_buffer, std::mutex *mutex)
{
    while (ring_buffer_num_items(ring_buffer, mutex) > 0) {
        AVFrame *frame;

        ring_buffer_dequeue(ring_buffer, &frame, mutex);
        av_frame_free(&frame);
    }
}

static void worker_thread(BufferSink *sink, std::mutex *mutex)
{
    int ret;

    while (sink->ctx) {
        if (need_filters_reinit)
            break;

        if (ring_buffer_num_items(&sink->consume_frames, mutex) < 1) {
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
            if (end > start)
                sink->speed = 1000000. * av_q2d(av_inv_q(sink->frame_rate)) / (end - start);
            if (ret < 0 && ret != AVERROR(EAGAIN))
                break;

            ring_buffer_enqueue(&sink->consume_frames, filter_frame, mutex);
        }

        if (paused)
            av_usleep(10000);
    }

    clear_ring_buffer(&sink->consume_frames, mutex);
    clear_ring_buffer(&sink->render_frames, mutex);
    clear_ring_buffer(&sink->purge_frames, mutex);
}

static int filters_setup()
{
    const AVFilter *new_filter;
    int ret;

    if (need_filters_reinit == false)
        return 0;

    for (unsigned i = 0; i < audio_sink_threads.size(); i++) {
        if (audio_sink_threads[i].joinable())
            audio_sink_threads[i].join();

        alDeleteSources(1, &abuffer_sinks[i].source);
        alDeleteBuffers(AL_BUFFERS, abuffer_sinks[i].buffers);
    }

    for (unsigned i = 0; i < video_sink_threads.size(); i++) {
        if (video_sink_threads[i].joinable())
            video_sink_threads[i].join();

        glDeleteTextures(1, &buffer_sinks[i].texture);
    }

    audio_sink_threads.clear();
    video_sink_threads.clear();

    need_filters_reinit = false;

    if (filter_nodes.size() == 0)
        return 0;

    buffer_sinks.clear();
    abuffer_sinks.clear();
    mutexes.clear();
    amutexes.clear();

    filter_graph_is_valid = false;

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

        if (!strcmp(filter_ctx->filter->name, "buffersink")) {
            BufferSink new_sink;

            new_sink.ctx = filter_ctx;
            new_sink.have_window_pos = false;
            new_sink.fullscreen = false;
            new_sink.show_osd = false;
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
            new_sink.have_window_pos = false;
            new_sink.fullscreen = false;
            new_sink.show_osd = false;
            new_sink.upscale_interpolator = 0;
            new_sink.downscale_interpolator = 0;
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
        ret = av_opt_serialize(filter_nodes[i].probe, 0, AV_OPT_SERIALIZE_SKIP_DEFAULTS,
                               &filter_nodes[i].ctx_options, '=', ':');
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Cannot serialize filter ctx options.\n");
            goto error;
        }

        av_freep(&filter_nodes[i].filter_options);
        ret = av_opt_serialize(filter_nodes[i].probe->priv, AV_OPT_FLAG_FILTERING_PARAM, AV_OPT_SERIALIZE_SKIP_DEFAULTS,
                               &filter_nodes[i].filter_options, '=', ':');
        if (ret < 0)
            av_log(NULL, AV_LOG_WARNING, "Cannot serialize filter private options.\n");

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
        unsigned x = edge2pad[p.first].node;
        unsigned y = edge2pad[p.second].node;
        unsigned x_pad = edge2pad[p.first].pad_index;
        unsigned y_pad = edge2pad[p.second].pad_index;

        if (!edge2pad[p.first].is_output)
            ;

        if (x >= filter_nodes.size() || y >= filter_nodes.size()) {
            av_log(NULL, AV_LOG_ERROR, "Cannot link filters: %s(%d) <-> %s(%d), index (%d,%d) out of range (%ld,%ld)\n",
                   filter_nodes[x].filter_label, x_pad, filter_nodes[y].filter_label, y_pad, x, y, filter_nodes.size(), filter_nodes.size());
            ret = AVERROR(EINVAL);
            goto error;
        }

        if ((ret = avfilter_link(filter_nodes[x].ctx, x_pad, filter_nodes[y].ctx, y_pad)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "Cannot link filters: %s(%d) <-> %s(%d)\n",
                   filter_nodes[x].filter_label, x_pad, filter_nodes[y].filter_label, y_pad);
            goto error;
        }
    }

    if ((ret = avfilter_graph_config(filter_graph, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot configure graph.\n");
        goto error;
    }

    filter_graph_is_valid = true;

    graphdump_text = avfilter_graph_dump(filter_graph, NULL);

    show_abuffersink_window = true;
    show_buffersink_window = true;
    show_dumpgraph_window = true;

error:

    if (ret < 0)
        return ret;

    std::vector<std::mutex> mutex_list(buffer_sinks.size());
    mutexes.swap(mutex_list);

    std::vector<std::thread> thread_list(buffer_sinks.size());
    video_sink_threads.swap(thread_list);

    for (unsigned i = 0; i < buffer_sinks.size(); i++) {
        buffer_sinks[i].id = i;
        buffer_sinks[i].label = av_asprintf("Video FilterGraph Output %d", i);
        buffer_sinks[i].time_base = av_buffersink_get_time_base(buffer_sinks[i].ctx);
        buffer_sinks[i].frame_rate = av_buffersink_get_frame_rate(buffer_sinks[i].ctx);
        buffer_sinks[i].pts = AV_NOPTS_VALUE;
        buffer_sinks[i].sample_index = 0;
        buffer_sinks[i].samples = NULL;
        buffer_sinks[i].nb_samples = 0;
        buffer_sinks[i].render_ring_size = 2;
        ring_buffer_init(&buffer_sinks[i].consume_frames);
        ring_buffer_init(&buffer_sinks[i].render_frames);
        ring_buffer_init(&buffer_sinks[i].purge_frames);

        glGenTextures(1, &buffer_sinks[i].texture);

        std::thread sink_thread(worker_thread, &buffer_sinks[i], &mutexes[i]);

        video_sink_threads[i].swap(sink_thread);
    }

    std::vector<std::mutex> amutex_list(abuffer_sinks.size());
    amutexes.swap(amutex_list);

    std::vector<std::thread> athread_list(abuffer_sinks.size());
    audio_sink_threads.swap(athread_list);

    for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
        abuffer_sinks[i].id = i;
        abuffer_sinks[i].label = av_asprintf("Audio FilterGraph Output %d", i);
        abuffer_sinks[i].time_base = av_buffersink_get_time_base(abuffer_sinks[i].ctx);
        abuffer_sinks[i].frame_rate = av_make_q(av_buffersink_get_sample_rate(abuffer_sinks[i].ctx), 1);
        abuffer_sinks[i].sample_index = 0;
        abuffer_sinks[i].nb_samples = 512;
        abuffer_sinks[i].pts = AV_NOPTS_VALUE;
        abuffer_sinks[i].samples = (float *)av_calloc(abuffer_sinks[i].nb_samples, sizeof(float));
        abuffer_sinks[i].render_ring_size = 1;
        ring_buffer_init(&abuffer_sinks[i].consume_frames);
        ring_buffer_init(&abuffer_sinks[i].render_frames);
        ring_buffer_init(&abuffer_sinks[i].purge_frames);

        abuffer_sinks[i].format = AL_FORMAT_MONO_FLOAT32;

        alGenBuffers(AL_BUFFERS, abuffer_sinks[i].buffers);
        for (unsigned j = 0; j < AL_BUFFERS; j++)
            abuffer_sinks[i].unprocessed_bufids.push_back(abuffer_sinks[i].buffers[j]);

        alGenSources(1, &abuffer_sinks[i].source);
        alSource3i(abuffer_sinks[i].source, AL_POSITION, 0, 0, -1);
        alSourcei(abuffer_sinks[i].source, AL_SOURCE_RELATIVE, AL_TRUE);
        alSourcei(abuffer_sinks[i].source, AL_ROLLOFF_FACTOR, 0);

        std::thread asink_thread(worker_thread, &abuffer_sinks[i], &amutexes[i]);

        audio_sink_threads[i].swap(asink_thread);
    }

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

    ImGui::SetNextWindowPos(ImGui::GetMousePos());
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
    ImGui::Text("Jump to Filter Commands Window:");
    ImGui::SameLine(align);
    ImGui::Text("F3");
    ImGui::Separator();
    ImGui::Text("Jump to FilterGraph Dump Window:");
    ImGui::SameLine(align);
    ImGui::Text("F4");
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
    ImGui::Text("Framestep forward:");
    ImGui::SameLine(align);
    ImGui::Text("'.'");
    ImGui::Separator();
    ImGui::Text("Jump to #numbered Video output:");
    ImGui::SameLine(align);
    ImGui::Text("Ctrl + <number>");
    ImGui::Separator();
    ImGui::Text("Jump to #numbered Audio output:");
    ImGui::SameLine(align);
    ImGui::Text("Alt + <number>");
    ImGui::Separator();
    ImGui::Text("Exit from output:");
    ImGui::SameLine(align);
    ImGui::Text("Shift + Q");
    ImGui::Separator();
    ImGui::End();
}

static void draw_osd(bool *p_open, int64_t pts, BufferSink *sink)
{
    const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration |
                                          ImGuiWindowFlags_AlwaysAutoResize |
                                          ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoNav |
                                          ImGuiWindowFlags_NoMouseInputs |
                                          ImGuiWindowFlags_NoFocusOnAppearing |
                                          ImGuiWindowFlags_NoMove;
    const int corner = 0;
    const float PAD_X = 10.0f;
    const float PAD_Y = 20.0f;

    ImVec2 work_pos = ImGui::GetWindowPos();
    ImVec2 work_size = ImGui::GetWindowSize();
    ImVec2 window_pos, window_pos_pivot;
    window_pos.x = (corner & 1) ? (work_pos.x + work_size.x - PAD_X) : (work_pos.x + PAD_X);
    window_pos.y = (corner & 2) ? (work_pos.y + work_size.y - PAD_Y) : (work_pos.y + PAD_Y);
    window_pos_pivot.x = (corner & 1) ? 1.0f : 0.0f;
    window_pos_pivot.y = (corner & 2) ? 1.0f : 0.0f;
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
    ImGui::SetNextWindowBgAlpha(0.77f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

    if (!ImGui::Begin("##OSD", p_open, window_flags)) {
        ImGui::End();
        ImGui::PopStyleVar();
        return;
    }

    ImGui::Text("TIME: %.5f", av_q2d(sink->time_base) * pts);
    ImGui::SameLine();
    ImGui::Text("SPEED: %.5f", sink->speed);
    ImGui::SameLine();
    ImGui::Text("FPS: %d/%d (%.5f)", sink->frame_rate.num,
                sink->frame_rate.den, av_q2d(sink->frame_rate));
    ImGui::End();
    ImGui::PopStyleVar();
}

static void draw_frame(GLuint *texture, bool *p_open, AVFrame *new_frame,
                       BufferSink *sink)
{
    ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
    int width, height, style = 0;

    if (!*p_open || !new_frame)
        return;

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
        style = 1;
    } else {
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
        ImGui::End();
        return;
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

    if (style)
        ImGui::PopStyleVar();

    if (sink->show_osd)
        draw_osd(&sink->show_osd, new_frame->pts, sink);

    if (ImGui::IsItemHovered() && ImGui::IsKeyDown(ImGuiKey_Z)) {
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImGui::BeginTooltip();
        float my_tex_w = (float)width;
        float my_tex_h = (float)height;
        ImVec4 tint_col   = ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // No tint
        ImVec4 border_col = ImVec4(1.0f, 1.0f, 1.0f, 0.5f); // 50% opaque white
        float region_sz = 32.0f;
        float region_x = io.MousePos.x - pos.x - region_sz * 0.5f;
        float region_y = io.MousePos.y - pos.y - region_sz * 0.5f;
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
    ImGui::End();
}

static void draw_aframe(bool *p_open, BufferSink *sink)
{
    ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
    char overlay[256] = { 0 };

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
        framestep = ImGui::IsKeyPressed(ImGuiKey_Period, true);
        if (framestep)
            paused = true;
        if (ImGui::IsKeyDown(ImGuiKey_Q) && ImGui::GetIO().KeyShift) {
            show_abuffersink_window = false;
            show_buffersink_window = false;
            filter_graph_is_valid = false;
        }
    }

    if (ImGui::IsKeyDown(ImGuiKey_0 + sink->id) && ImGui::GetIO().KeyAlt)
        focus_abuffersink_window = sink->id;

    snprintf(overlay, sizeof(overlay), "TIME: %.5f\nSPEED: %f", sink->pts != AV_NOPTS_VALUE ? av_q2d(sink->time_base) * sink->pts : NAN, sink->speed);
    ImGui::PlotLines("Audio Samples", sink->samples, sink->nb_samples, 0, overlay, -1.0f, 1.0f, ImVec2(0, 80.0f));

    ImGui::End();
}

static void play_sound(AVFrame *frame, BufferSink *sink)
{
    ALint processed, state, queued;

    alGetSourcei(sink->source, AL_BUFFERS_PROCESSED, &processed);
    while (processed > 0) {
        ALuint bufid;

        alSourceUnqueueBuffers(sink->source, 1, &bufid);
        processed--;

        sink->processed_bufids.push_back(bufid);
    }

    alGetSourcei(sink->source, AL_SOURCE_STATE, &state);
    alGetSourcei(sink->source, AL_BUFFERS_QUEUED, &queued);
    if (queued >= 1 && state != AL_PLAYING)
        alSourcePlay(sink->source);
    else if (queued < 1 && state != AL_PAUSED)
        alSourcePause(sink->source);

    if (sink->processed_bufids.size() > 0 && frame->nb_samples > 0) {
        ALuint bufid = sink->processed_bufids.back();

        sink->processed_bufids.pop_back();
        alBufferData(bufid, sink->format, frame->extended_data[0],
                     (ALsizei)frame->nb_samples * sizeof(float), frame->sample_rate);
        alSourceQueueBuffers(sink->source, 1, &bufid);
    } else if (sink->unprocessed_bufids.size() > 0 && frame->nb_samples > 0) {
        ALuint bufid = sink->unprocessed_bufids.back();

        sink->unprocessed_bufids.pop_back();
        alBufferData(bufid, sink->format, frame->extended_data[0],
                     (ALsizei)frame->nb_samples * sizeof(float), frame->sample_rate);
        alSourceQueueBuffers(sink->source, 1, &bufid);
    }

    alGetSourcei(sink->source, AL_SOURCE_STATE, &state);
    alGetSourcei(sink->source, AL_BUFFERS_QUEUED, &queued);

    if (queued >= 1 && state != AL_PLAYING)
        alSourcePlay(sink->source);
    else if (queued < 1 && state != AL_PAUSED)
        alSourcePause(sink->source);

    sink->need_more = queued < (AL_BUFFERS / 2);
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
    FilterNode node;

    if (ImGui::MenuItem(filter->name)) {
        node.filter = filter;
        node.id = filter_nodes.size();
        node.filter_name = av_strdup(filter->name);
        node.filter_label = av_asprintf("%s%d", filter->name, node.id);
        node.filter_options = NULL;
        node.ctx_options = NULL;
        node.probe_graph = NULL;
        node.probe = NULL;
        node.ctx = NULL;
        node.pos = click_pos;
        node.colapsed = false;
        node.set_pos = true;

        filter_nodes.push_back(node);
    }

    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", filter->description);
}

static void draw_options(FilterNode *node, void *av_class, bool filter_private)
{
    const AVOption *opt = NULL;
    const void *obj = (const void *)(filter_private ? &node->filter->priv_class : (const void *)node->probe);
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
                    if (ImGui::DragScalar(opt->name, ImGuiDataType_Double, &dvalue, 1, &min, &max, "%f", ImGuiSliderFlags_AlwaysClamp)) {
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
                        if (ImGui::BeginCombo("##const flags values", 0, 0)) {
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
                        if (ImGui::BeginCombo("##const int values", 0, 0)) {
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
                    static AVRational rate = (AVRational){ 0, 0 };
                    int irate[2] = { 0, 0 };
                    char rate_str[256];

                    if (rate.num == 0 && rate.den == 0) {
                        if (av_opt_get_video_rate(av_class, opt->name, 0, &rate))
                            av_parse_video_rate(&rate, opt->default_val.str);
                    }
                    irate[0] = rate.num;
                    irate[1] = rate.den;
                    if (ImGui::DragInt2(opt->name, irate, 1, -8192, 8192)) {
                        rate.num = irate[0];
                        rate.den = irate[1];
                        if (av_opt_set_video_rate(av_class, opt->name, rate, 0)) {
                            snprintf(rate_str, sizeof(rate_str), "%d/%d", rate.num, rate.den);
                            av_opt_set(av_class, opt->name, rate_str, 0);
                        }
                    }
                }
                break;
            case AV_OPT_TYPE_PIXEL_FMT:
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

static void draw_node_options(FilterNode *node)
{
    AVFilterContext *probe_ctx;
    AVFilterGraph *probe_graph;
    void *av_class;

    if (!node->probe_graph)
        node->probe_graph = avfilter_graph_alloc();
    probe_graph = node->probe_graph;
    if (!probe_graph)
        return;
    probe_graph->nb_threads = 1;

    if (!node->probe)
        node->probe = avfilter_graph_alloc_filter(probe_graph, node->filter, "probe");
    probe_ctx = node->probe;
    if (!probe_ctx)
        return;

    av_class = probe_ctx->priv;
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
            if (!node->probe_graph)
                avfilter_free(node->probe);
            avfilter_graph_free(&node->probe_graph);
            node->probe = NULL;
            avfilter_free(node->ctx);
            node->ctx = NULL;
            node->colapsed = false;
            filter_nodes.erase(filter_nodes.begin() + node->id);
            return;
        }
    }

    if (!ImGui::BeginListBox("##List of Filter Options"))
        return;
    draw_options(node, av_class, 1);
    ImGui::NewLine();
    draw_options(node, probe_ctx, 0);

    ImGui::EndListBox();
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

static void show_filtergraph_editor(bool *p_open, bool focused)
{
    int edge;

    if (focused)
        ImGui::SetNextWindowFocus();
    ImGui::SetNextWindowSizeConstraints(ImVec2(600, 500), ImVec2(display_w, display_h), NULL);
    if (!ImGui::Begin("FilterGraph Editor", p_open, 0)) {
        ImGui::End();
        return;
    }

    ImNodes::EditorContextSet(node_editor_context);

    edge2pad.clear();
    edge2type.clear();

    ImNodes::BeginNodeEditor();

    if (ImGui::IsKeyReleased(ImGuiKey_Enter) && ImGui::GetIO().KeyCtrl)
        need_filters_reinit = true;

    const bool open_popup = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
        ImNodes::IsEditorHovered() && ((ImGui::IsKeyReleased(ImGuiKey_A) && !ImGui::GetIO().KeyShift) ||
        ImGui::IsMouseReleased(ImGuiMouseButton_Right));

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.f, 8.f));
    if (!ImGui::IsAnyItemHovered() && open_popup) {
        ImGui::OpenPopup("Add Filter");
    }

    if (ImGui::BeginPopup("Add Filter")) {
        const ImVec2 click_pos = ImGui::GetMousePosOnOpeningCurrentPopup();

        if (ImGui::BeginMenu("Source Filters")) {
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
        if (ImGui::BeginMenu("Simple Filters")) {
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

        if (ImGui::BeginMenu("Complex Filters")) {
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

        if (ImGui::BeginMenu("Media Filters")) {
            const AVFilter *filter = NULL;
            void *iterator = NULL;

            while ((filter = av_filter_iterate(&iterator))) {
                if (!is_media_filter(filter))
                    continue;

                handle_nodeitem(filter, click_pos);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Sink Filters")) {
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
                ImGui::InputInt("Max Number of FilterGraph Threads", &filter_graph_nb_threads);
                ImGui::InputInt("Auto Conversion Type for FilterGraph", &filter_graph_auto_convert_flags);
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

        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();

    edge = 0;
    for (unsigned i = 0; i < filter_nodes.size(); i++) {
        FilterNode *filter_node = &filter_nodes[i];

        filter_node->edge = edge;
        edge2type.push_back(std::make_pair(edge, AVMEDIA_TYPE_UNKNOWN));
        edge2pad.push_back(Edge2Pad { i, 0, 0 });
        ImNodes::BeginNode(edge++);
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

        for (unsigned j = 0; j < filter_ctx->nb_inputs; j++) {
            enum AVMediaType media_type;

            media_type = avfilter_pad_get_type(filter_ctx->input_pads, j);
            if (media_type == AVMEDIA_TYPE_VIDEO) {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(  0, 255, 255, 255));
            } else {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(255, 255,   0, 255));
            }
            edge2type.push_back(std::make_pair(edge, media_type));
            edge2pad.push_back(Edge2Pad { i, false, j });
            ImNodes::BeginInputAttribute(edge++);
            ImGui::Text("%s", avfilter_pad_get_name(filter_ctx->input_pads, j));
            ImNodes::EndInputAttribute();
            ImNodes::PopColorStyle();
        }

        for (unsigned j = 0; j < filter_ctx->nb_outputs; j++) {
            enum AVMediaType media_type;

            media_type = avfilter_pad_get_type(filter_ctx->output_pads, j);
            if (media_type == AVMEDIA_TYPE_VIDEO) {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(  0, 255, 255, 255));
            } else {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(255, 255,   0, 255));
            }
            edge2type.push_back(std::make_pair(edge, media_type));
            edge2pad.push_back(Edge2Pad { i, true, j });
            ImNodes::BeginOutputAttribute(edge++);
            ImGui::Text("%s", avfilter_pad_get_name(filter_ctx->output_pads, j));
            ImNodes::EndOutputAttribute();
            ImNodes::PopColorStyle();
        }

        ImNodes::EndNode();
        ImNodes::SetNodeDraggable(filter_node->edge, true);
    }

    for (unsigned i = 0; i < filter_links.size(); i++) {
        const std::pair<int, int> p = filter_links[i];
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

    int start_attr, end_attr;
    if (ImNodes::IsLinkCreated(&start_attr, &end_attr)) {
        enum AVMediaType first  = edge2type[start_attr].second;
        enum AVMediaType second = edge2type[end_attr].second;

        if (first == second)
            filter_links.push_back(std::make_pair(start_attr, end_attr));
    }

    int link_id;
    if (ImNodes::IsLinkDestroyed(&link_id))
        filter_links.erase(filter_links.begin() + link_id);

    const int links_selected = ImNodes::NumSelectedLinks();
    if (links_selected > 0 && ImGui::IsKeyReleased(ImGuiKey_X)) {
        static std::vector<int> selected_links;

        selected_links.resize(static_cast<size_t>(links_selected));
        ImNodes::GetSelectedLinks(selected_links.data());

        for (const int edge_id : selected_links) {
            if (filter_links.size() == 0)
                break;
            filter_links.erase(filter_links.begin() + edge_id);
        }
    }

    const int nodes_selected = ImNodes::NumSelectedNodes();
    if (nodes_selected > 0 && ImGui::IsKeyReleased(ImGuiKey_X) && ImGui::GetIO().KeyShift) {
        static std::vector<int> selected_nodes;

        selected_nodes.resize(static_cast<size_t>(nodes_selected));
        ImNodes::GetSelectedNodes(selected_nodes.data());
        for (const int node_id : selected_nodes) {
            const unsigned node = edge2pad[node_id].node;

            filter_nodes[node].filter = NULL;
            avfilter_free(filter_nodes[node].ctx);
            av_freep(&filter_nodes[node].filter_name);
            av_freep(&filter_nodes[node].filter_label);
            av_freep(&filter_nodes[node].filter_options);
            av_freep(&filter_nodes[node].ctx_options);
            if (!filter_nodes[node].probe_graph)
                avfilter_free(filter_nodes[node].probe);
            avfilter_graph_free(&filter_nodes[node].probe_graph);
            filter_nodes[node].probe = NULL;
        }
    }

    bool erased = false;
    if (filter_nodes.size() > 0) {
        unsigned i = filter_nodes.size() - 1;
        do {
            if (!filter_nodes[i].filter) {
                const int removed_edge = filter_nodes[i].edge;
                const int subtract = removed_edge - (i > 0 ? filter_nodes[i-1].edge : 0);

                filter_nodes.erase(filter_nodes.begin() + i);
                erased = true;

                if (filter_links.size() > 0) {
                    unsigned l = filter_links.size() - 1;
                    do {
                        const std::pair<int, int> p = filter_links[l];
                        int a = p.first;
                        int b = p.second;

                        if (a == removed_edge || b == removed_edge) {
                            filter_links.erase(filter_links.begin() + l);
                        } else {
                            if (a > removed_edge)
                                a -= subtract;
                            if (b > removed_edge)
                                b -= subtract;

                            filter_links[l] = std::make_pair(a, b);
                        }
                    } while (l--);
                }
            }
        } while (i--);
    }

    if (erased && filter_nodes.size() > 0) {
        unsigned i = filter_nodes.size() - 1;
        do {
            filter_nodes[i].set_pos = true;
        } while (i--);
    }

    if (ImGui::IsKeyReleased(ImGuiKey_A) && ImGui::GetIO().KeyShift) {
        const AVFilter *buffersink  = avfilter_get_by_name("buffersink");
        const AVFilter *abuffersink = avfilter_get_by_name("abuffersink");
        std::vector<int> unconnected_edges;
        std::vector<int> connected_edges;

        for (unsigned l = 0; l < filter_links.size(); l++) {
            const std::pair<int, int> p = filter_links[l];

            connected_edges.push_back(p.first);
            connected_edges.push_back(p.second);
        }

        for (int e = 0; e < edge; e++) {
            unsigned ee = 0;
            for (; ee < connected_edges.size(); ee++) {
                if (e == connected_edges[ee])
                    break;
            }

            if (ee == connected_edges.size() &&
                edge2pad[e].is_output == true)
                unconnected_edges.push_back(e);
        }

        for (unsigned i = 0; i < unconnected_edges.size(); i++) {
            const int e = unconnected_edges[i];
            enum AVMediaType type = edge2type[e].second;
            FilterNode node;

            node.filter = type == AVMEDIA_TYPE_AUDIO ? abuffersink : buffersink;
            node.id = filter_nodes.size();
            node.filter_name = av_strdup(node.filter->name);
            node.filter_label = av_asprintf("%s%d", node.filter->name, node.id);
            node.filter_options = NULL;
            node.ctx_options = NULL;
            node.probe_graph = NULL;
            node.probe = NULL;
            node.ctx = NULL;
            node.pos = ImVec2(filter_nodes[edge2pad[e].node].pos.x + 100 * (i + 1), filter_nodes[edge2pad[e].node].pos.y + 100 * (i + 1));
            node.colapsed = false;
            node.set_pos = true;
            node.edge = edge++;

            filter_nodes.push_back(node);
            filter_links.push_back(std::make_pair(e, edge++));
        }
    }

    ImGui::End();
}

static void show_commands(bool *p_open, bool focused)
{
    static unsigned selected_filter = -1;
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
    if (ImGui::BeginListBox("##Filters", ImVec2(400, 300))) {
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

            if (is_opened && selected_filter == n) {
                if (ctx->filter->process_command) {
                    if (ImGui::TreeNode("Commands")) {
                        std::vector<OptStorage> opt_storage = filter_nodes[n].opt_storage;
                        const AVOption *opt = NULL;
                        unsigned opt_index = 0;

                        if (is_opened && clean_storage) {
                            opt_storage.clear();
                            clean_storage = false;
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
                                            memcpy(string, opt_storage[opt_index].u.str, FFMIN(sizeof(string) - 1, strlen(opt_storage[opt_index].u.str)));
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

                        ImGui::TreePop();
                    }
                }

                if (ctx->filter->flags & AVFILTER_FLAG_SUPPORT_TIMELINE) {
                    if (ImGui::TreeNode("Timeline")) {
                        if (ImGui::Button(ctx->is_disabled ? "Enable" : "Disable")) {
                            toggle_filter = n;
                        }
                        ImGui::TreePop();
                    }
                }
            }
        }

        ImGui::EndListBox();
    }
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
    alListenerfv(AL_POSITION, position);
    alListenerfv(AL_ORIENTATION, direction);

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

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(width, height, "lavfi-preview", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

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
    //ImGui::StyleColorsClassic();

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
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        if (show_abuffersink_window == false) {
            if (audio_sink_threads.size() > 0) {
                need_filters_reinit = true;

                for (unsigned i = 0; i < audio_sink_threads.size(); i++) {
                    if (audio_sink_threads[i].joinable())
                        audio_sink_threads[i].join();

                    av_freep(&abuffer_sinks[i].label);
                    av_freep(&abuffer_sinks[i].samples);
                    alDeleteSources(1, &abuffer_sinks[i].source);
                    alDeleteBuffers(AL_BUFFERS, abuffer_sinks[i].buffers);
                }

                audio_sink_threads.clear();

                need_filters_reinit = false;
            }
        }

        if (show_buffersink_window == false) {
            if (video_sink_threads.size() > 0) {
                need_filters_reinit = true;

                for (unsigned i = 0; i < video_sink_threads.size(); i++) {
                    if (video_sink_threads[i].joinable())
                        video_sink_threads[i].join();

                    av_freep(&buffer_sinks[i].label);
                    glDeleteTextures(1, &buffer_sinks[i].texture);
                }

                video_sink_threads.clear();

                need_filters_reinit = false;
            }
        }

        filters_setup();

        if (mutexes.size() == buffer_sinks.size()) {
            for (unsigned i = 0; i < buffer_sinks.size(); i++) {
                BufferSink *sink = &buffer_sinks[i];
                AVFrame *render_frame = NULL;

                if (ring_buffer_num_items(&sink->render_frames, &mutexes[i]) > sink->render_ring_size - 1)
                    continue;

                ring_buffer_dequeue(&sink->consume_frames, &render_frame, &mutexes[i]);
                if (!render_frame)
                    continue;
                ring_buffer_enqueue(&sink->render_frames, render_frame, &mutexes[i]);
            }
        }

        if (amutexes.size() == abuffer_sinks.size()) {
            for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
                BufferSink *sink = &abuffer_sinks[i];
                AVFrame *render_frame = NULL;

                if (ring_buffer_num_items(&sink->render_frames, &amutexes[i]) > sink->render_ring_size - 1)
                    continue;

                ring_buffer_dequeue(&sink->consume_frames, &render_frame, &amutexes[i]);
                if (!render_frame)
                    continue;
                ring_buffer_enqueue(&sink->render_frames, render_frame, &amutexes[i]);
            }
        }

        if (amutexes.size() == abuffer_sinks.size() && show_abuffersink_window == true) {
            for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
                BufferSink *sink = &abuffer_sinks[i];
                AVFrame *play_frame = NULL;

                ring_buffer_peek(&sink->render_frames, &play_frame, 0, &amutexes[i]);
                if (play_frame) {
                    const float *src = (const float *)play_frame->extended_data[0];

                    if (src) {
                        sink->pts = play_frame->pts;
                        sink->samples[sink->sample_index++] = src[0];
                        if (sink->sample_index >= sink->nb_samples)
                            sink->sample_index = 0;
                    }

                    if (paused == false || framestep) {
                        play_sound(play_frame, sink);
                        play_frame->nb_samples = 0;
                    }
                }
            }
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (mutexes.size() == buffer_sinks.size() && show_buffersink_window == true) {
            for (unsigned i = 0; i < buffer_sinks.size(); i++) {
                BufferSink *sink = &buffer_sinks[i];
                AVFrame *render_frame = NULL;

                ring_buffer_peek(&sink->render_frames, &render_frame, 0, &mutexes[i]);
                if (!render_frame)
                    continue;

                draw_frame(&sink->texture, &show_buffersink_window, render_frame, sink);
            }
        }

        if (amutexes.size() == abuffer_sinks.size() && show_abuffersink_window == true) {
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
        focused = ImGui::IsKeyReleased(ImGuiKey_F2);
        if (focused)
            show_filtergraph_editor_window = true;
        if (show_filtergraph_editor_window)
            show_filtergraph_editor(&show_filtergraph_editor_window, focused);
        show_help = ImGui::IsKeyDown(ImGuiKey_F1);
        if (show_help)
            draw_help(&show_help);

        // Rendering
        ImGui::Render();
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        if (mutexes.size() == buffer_sinks.size()) {
            for (unsigned i = 0; i < buffer_sinks.size(); i++) {
                BufferSink *sink = &buffer_sinks[i];

                if (!paused || framestep) {
                    AVFrame *purge_frame = NULL;

                    if (ring_buffer_num_items(&sink->render_frames, &mutexes[i]) < sink->render_ring_size)
                        continue;

                    ring_buffer_dequeue(&sink->render_frames, &purge_frame, &mutexes[i]);
                    if (!purge_frame)
                        continue;
                    ring_buffer_enqueue(&sink->purge_frames, purge_frame, &mutexes[i]);
                }

                clear_ring_buffer(&sink->purge_frames, &mutexes[i]);
            }
        }

        if (amutexes.size() == abuffer_sinks.size()) {
            for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
                BufferSink *sink = &abuffer_sinks[i];

                if ((!paused || framestep) && sink->need_more) {
                    AVFrame *purge_frame = NULL;

                    if (ring_buffer_num_items(&sink->render_frames, &amutexes[i]) < sink->render_ring_size)
                        continue;

                    ring_buffer_dequeue(&sink->render_frames, &purge_frame, &amutexes[i]);
                    if (!purge_frame)
                        continue;
                    ring_buffer_enqueue(&sink->purge_frames, purge_frame, &amutexes[i]);
                }

                clear_ring_buffer(&sink->purge_frames, &amutexes[i]);
            }
        }
    }

    need_filters_reinit = true;
    for (unsigned i = 0; i < video_sink_threads.size(); i++) {
        if (video_sink_threads[i].joinable())
            video_sink_threads[i].join();

        av_freep(&buffer_sinks[i].label);
        glDeleteTextures(1, &buffer_sinks[i].texture);
    }

    for (unsigned i = 0; i < audio_sink_threads.size(); i++) {
        if (audio_sink_threads[i].joinable())
            audio_sink_threads[i].join();

        av_freep(&abuffer_sinks[i].label);
        av_freep(&abuffer_sinks[i].samples);
        alDeleteSources(1, &abuffer_sinks[i].source);
        alDeleteBuffers(AL_BUFFERS, abuffer_sinks[i].buffers);
    }

    video_sink_threads.clear();
    audio_sink_threads.clear();

    for (unsigned i = 0; i < filter_nodes.size(); i++) {
        FilterNode *node = &filter_nodes[i];

        av_freep(&node->filter_name);
        av_freep(&node->filter_label);
        if (!node->probe_graph)
            avfilter_free(node->probe);
        avfilter_graph_free(&node->probe_graph);
        node->probe = NULL;
        node->ctx = NULL;
    }

    filter_nodes.clear();

    av_freep(&graphdump_text);

    avfilter_graph_free(&filter_graph);
    filter_links.clear();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImNodes::EditorContextFree(node_editor_context);
    node_editor_context = NULL;
    ImNodes::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    alcDestroyContext(al_ctx);
    al_ctx = NULL;
    alcCloseDevice(al_dev);
    al_dev = NULL;

    return 0;
}
