// Dear ImGui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)
// If you are new to Dear ImGui, read documentation from the docs/ folder + read the top of imgui.cpp.
// Read online: https://github.com/ocornut/imgui/tree/master/docs

#include <queue>
#include <thread>
#include <mutex>

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers
#include <unistd.h>

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/avstring.h>
#include <libavutil/dict.h>
#include <libavutil/opt.h>
#include <libavutil/parseutils.h>
#include <libavutil/time.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavformat/avformat.h>
}

typedef struct ValueStorage {
    int inited;
    union {
        int i32;
        float flt;
        int64_t i64;
        uint64_t u64;
        double dbl;
        AVRational q;
        char *str;
    } u;
} ValueStorage;

typedef struct FiltersOptions {
    char *filter_name;
    char *filter_label;
    char *ctx_options;
    char *filter_options;
    ValueStorage value_storage[64];
} FiltersOptions;

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

double speed = 1.f;

bool show_filters_list_window = true;
bool show_buffersink_window = true;
bool show_dumpgraph_window = true;
bool show_commands_window = true;
bool show_osd = true;

bool framestep = false;
bool paused = false;
int width = 1280;
int height = 720;
int need_filters_reinit = 1;
FiltersOptions filters_options[1024] = { { NULL, NULL, NULL, NULL, { 0, 0 } } };
AVFilterContext *new_filters[1024] = { NULL };
int nb_all_filters = 0;
AVFilterContext *buffersink_ctx = NULL;
AVRational buffersink_time_base;
AVRational buffersink_frame_rate;
AVFilterContext *filter_ctx = NULL;
AVFilterContext *probe_ctx = NULL;
AVFilterGraph *filter_graph = NULL;
AVFilterGraph *probe_graph = NULL;
AVFilterInOut *outputs = NULL;
AVFilterInOut *inputs = NULL;
char *graphdump_text = NULL;
GLuint frame_texture;

std::queue<AVFrame *> filter_frames;
std::thread video_sink_thread;
std::mutex filter_frames_mutex;

const enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_RGB0, AV_PIX_FMT_NONE };

static void worker_thread(AVFilterContext *ctx, AVRational rate, std::queue<AVFrame *> *frames, std::mutex *mutex)
{
    int ret;

    while (ctx) {
        if (need_filters_reinit)
            break;

        if (!paused || framestep) {
            mutex->lock();
            if (frames->size() <= 2) {
                AVFrame *filter_frame;
                int64_t start, end;

                mutex->unlock();
                filter_frame = av_frame_alloc();
                start = av_gettime_relative();
                ret = av_buffersink_get_frame_flags(ctx, filter_frame, 0);
                end = av_gettime_relative();
                if (end > start)
                    speed = 1000000. * av_q2d(av_inv_q(rate)) / (end - start);
                if (ret < 0 && ret != AVERROR(EAGAIN))
                    break;

                mutex->lock();
                frames->push(filter_frame);
                framestep = false;
            }

            while (frames->size() > 2) {
                AVFrame *pop_frame = frames->front();
                frames->pop();
                av_frame_free(&pop_frame);
            }
            mutex->unlock();
        } else {
            usleep(10000);
        }
    }

    mutex->lock();
    while (frames->size() > 0) {
        AVFrame *pop_frame = frames->front();

        frames->pop();
        av_frame_free(&pop_frame);
    }
    mutex->unlock();
}

static int filters_setup()
{
    const AVFilter *new_filter;
    const AVFilter *buffersink;
    int ret, i;

    if (need_filters_reinit == 0)
        return 0;

    if (video_sink_thread.joinable())
        video_sink_thread.join();

    if (nb_all_filters <= 0)
        return 0;
    if (!filters_options[0].filter_name)
        return 0;
    need_filters_reinit = 0;

    buffersink_ctx = NULL;

    av_freep(&graphdump_text);
    avfilter_inout_free(&outputs);
    avfilter_inout_free(&inputs);
    avfilter_graph_free(&filter_graph);

    buffersink = avfilter_get_by_name("buffersink");
    if (!buffersink) {
        av_log(NULL, AV_LOG_ERROR, "Cannot find buffersink\n");
        ret = AVERROR(ENOSYS);
        goto error;
    }

    outputs = avfilter_inout_alloc();
    inputs  = avfilter_inout_alloc();
    filter_graph = avfilter_graph_alloc();
    if (!outputs || !inputs || !filter_graph) {
        av_log(NULL, AV_LOG_ERROR, "Cannot allocate graph\n");
        ret = AVERROR(ENOMEM);
        goto error;
    }

    ret = avfilter_graph_create_filter(&buffersink_ctx, buffersink, "out",
                                       NULL, NULL, filter_graph);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot create buffer sink\n");
        goto error;
    }

    ret = av_opt_set_int_list(buffersink_ctx, "pix_fmts", pix_fmts,
                              AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot set output pixel format\n");
        goto error;
    }

    for (i = 0; i < nb_all_filters; i++) {
        new_filter = avfilter_get_by_name(filters_options[i].filter_name);
        if (!new_filter) {
            av_log(NULL, AV_LOG_ERROR, "Cannot [%d] get filter by name: %s\n", i, filters_options[i].filter_name);
            ret = AVERROR(ENOSYS);
            goto error;
        }

        filter_ctx = avfilter_graph_alloc_filter(filter_graph, new_filter, filters_options[i].filter_label);
        if (!filter_ctx) {
            av_log(NULL, AV_LOG_ERROR, "Cannot allocate filter context\n");
            ret = AVERROR(ENOMEM);
            goto error;
        }

        av_opt_set_defaults(filter_ctx);
        new_filters[i] = filter_ctx;

        ret = av_opt_set_from_string(filter_ctx, filters_options[i].ctx_options, NULL, "=", ":");
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Error setting filter ctx options\n");
            goto error;
        }

        ret = av_opt_set_from_string(filter_ctx->priv, filters_options[i].filter_options, NULL, "=", ":");
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Error setting filter priv options\n");
            goto error;
        }

        ret = avfilter_init_str(filter_ctx, NULL);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Cannot init str for filter\n");
            goto error;
        }
    }

    if ((ret = avfilter_link(new_filters[nb_all_filters - 1], 0, buffersink_ctx, 0)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot link last filter with sink\n");
        goto error;
    }

    for (i = nb_all_filters - 1; i > 0; i--) {
        if ((ret = avfilter_link(new_filters[i-1], 0, new_filters[i], 0)) < 0) {
            av_log(NULL, AV_LOG_ERROR, "Cannot link filters\n");
            goto error;
        }
    }

    if ((ret = avfilter_graph_config(filter_graph, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot configure graph\n");
        goto error;
    }

    graphdump_text = avfilter_graph_dump(filter_graph, NULL);

    show_buffersink_window = true;
    show_dumpgraph_window = true;

error:

    if (ret < 0) {
        avfilter_inout_free(&outputs);
        avfilter_inout_free(&inputs);
        avfilter_graph_free(&filter_graph);
        buffersink_ctx = NULL;
        nb_all_filters = 0;

        return ret;
    }

    buffersink_time_base = av_buffersink_get_time_base(buffersink_ctx);
    buffersink_frame_rate = av_buffersink_get_frame_rate(buffersink_ctx);
    std::thread sink_thread(worker_thread, buffersink_ctx, buffersink_frame_rate, &filter_frames, &filter_frames_mutex);
    video_sink_thread.swap(sink_thread);

    return 0;
}

static bool load_frame(GLuint *out_texture, int *width, int *height, AVFrame *frame)
{
    if (!frame)
        return false;

    *width  = frame->width;
    *height = frame->height;

    glBindTexture(GL_TEXTURE_2D, *out_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, frame->linesize[0] / 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frame->width, frame->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, frame->data[0]);

    return true;
}

static void draw_osd(bool *p_open, int64_t pts)
{
    const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration |
                                          ImGuiWindowFlags_AlwaysAutoResize |
                                          ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoNav |
                                          ImGuiWindowFlags_NoMouseInputs |
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

    if (!ImGui::Begin("##OSD", p_open, window_flags)) {
        ImGui::End();
        return;
    }

    ImGui::BringWindowToDisplayFront(ImGui::GetCurrentWindow());
    if (pts == AV_NOPTS_VALUE)
        ImGui::TextColored(ImVec4(1.f, 1.f, 1.f, 0.8f), "TIME: ?.?? \t SPEED: %.5f", speed);
    else
        ImGui::TextColored(ImVec4(1.f, 1.f, 1.f, 0.8f), "TIME: %.5f \t SPEED: %.5f", av_q2d(buffersink_time_base) * pts, speed);
    ImGui::End();
}

static void draw_frame(int ret, GLuint *texture, bool *p_open, AVFrame *new_frame)
{
    int width, height;

    if (ret < 0 || !*p_open || !new_frame)
        return;
    ret = load_frame(texture, &width, &height, new_frame);
    if (ret) {
        if (!ImGui::Begin("filtergraph output", p_open, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::End();
            return;
        }

        if (ImGui::IsWindowFocused()) {
            if (ImGui::IsKeyReleased(ImGuiKey_Space))
                paused = !paused;
            framestep = ImGui::IsKeyPressed(ImGuiKey_Period, true);
            if (ImGui::IsKeyReleased(ImGuiKey_O))
                show_osd = !show_osd;
        }

        ImGui::Image((void*)(intptr_t)*texture, ImVec2(width, height));
        if (show_osd)
            draw_osd(&show_osd, new_frame->pts);

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
        avfilter_pad_get_type(filter->inputs, 0) == AVMEDIA_TYPE_AUDIO &&
        avfilter_pad_get_type(filter->inputs, 1) == AVMEDIA_TYPE_AUDIO) {
        return true;
    }
    return false;
}

static bool is_simple_video_filter(const AVFilter *filter)
{
    if (is_simple_filter(filter) &&
        avfilter_pad_get_type(filter->inputs, 0) == AVMEDIA_TYPE_VIDEO &&
        avfilter_pad_get_type(filter->inputs, 1) == AVMEDIA_TYPE_VIDEO) {
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

static void show_commands(bool *p_open)
{
    static unsigned selected_filter = -1;
    static unsigned toggle_filter = UINT_MAX;

    if (!filter_graph)
        return;

    if (!ImGui::Begin("Filters Commands", p_open, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        return;
    }
    if (ImGui::BeginListBox("##Filters", ImVec2(400, 300))) {
        static ImGuiTextFilter imgui_filter;

        imgui_filter.Draw();
        for (unsigned n = 0; n < filter_graph->nb_filters; n++) {
            const AVFilterContext *ctx = filter_graph->filters[n];
            const bool is_selected = selected_filter == n;
            static bool is_opened = false;
            static bool clean_storage = true;

            if (!imgui_filter.PassFilter(ctx->filter->name))
                continue;

            if (ImGui::Selectable(ctx->filter->name, is_selected)) {
                selected_filter = n;
            }

            if (ImGui::IsItemActive() || ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", ctx->name);
            }

            if (ImGui::IsItemClicked() || ImGui::IsItemActivated()) {
                selected_filter = n;
                is_opened = !is_opened;
            }

            if (is_opened && selected_filter == n) {
                if (ctx->filter->process_command) {
                    ValueStorage *value_storage = filters_options[n].value_storage;
                    if (ImGui::TreeNode("Commands")) {
                        const AVOption *opt = NULL;
                        static int pressed = -1;
                        int opt_index = 0;

                        if (is_opened && clean_storage) {
                            memset(value_storage, 0, sizeof(*value_storage));
                            clean_storage = 0;
                        }

                        while ((opt = av_opt_next(ctx->priv, opt))) {
                            double min, max;
                            void *ptr;

                            if (!(opt->flags & AV_OPT_FLAG_RUNTIME_PARAM)) {
                                opt_index++;
                                continue;
                            }

                            if (!query_ranges((void *)&ctx->filter->priv_class, opt, &min, &max)) {
                                opt_index++;
                                continue;
                            }

                            ptr = av_opt_ptr(ctx->filter->priv_class, ctx->priv, opt->name);
                            if (!ptr) {
                                opt_index++;
                                continue;
                            }

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
                                        pressed = opt_index;
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

                                        if (!value_storage[opt_index].inited) {
                                            value_storage[opt_index].u.i32 = *(int *)ptr;
                                            value_storage[opt_index].inited = 1;
                                        }
                                        value = value_storage[opt_index].u.i32;
                                        if (ImGui::SliderInt(opt->name, &value, imin, imax)) {
                                            value_storage[opt_index].u.i32 = value;
                                        }
                                    }
                                    break;
                                case AV_OPT_TYPE_INT:
                                    {
                                        int value = *(int *)ptr;
                                        int imin = min;
                                        int imax = max;

                                        if (!value_storage[opt_index].inited) {
                                            value_storage[opt_index].u.i32 = *(int *)ptr;
                                            value_storage[opt_index].inited = 1;
                                        }
                                        value = value_storage[opt_index].u.i32;
                                        if (imax < INT_MAX/2 && imin > INT_MIN/2) {
                                            if (ImGui::SliderInt(opt->name, &value, imin, imax)) {
                                                value_storage[opt_index].u.i32 = value;
                                            }
                                        } else {
                                            if (ImGui::DragInt(opt->name, &value, imin, imax, ImGuiSliderFlags_AlwaysClamp)) {
                                                value_storage[opt_index].u.i32 = value;
                                            }
                                        }
                                    }
                                    break;
                                case AV_OPT_TYPE_INT64:
                                    {
                                        int64_t value = *(int64_t *)ptr;
                                        int64_t imin = min;
                                        int64_t imax = max;

                                        if (!value_storage[opt_index].inited) {
                                            value_storage[opt_index].u.i64 = *(int64_t *)ptr;
                                            value_storage[opt_index].inited = 1;
                                        }
                                        value = value_storage[opt_index].u.i64;
                                        if (ImGui::DragScalar(opt->name, ImGuiDataType_S64, &value, 1, &imin, &imax, "%ld", ImGuiSliderFlags_AlwaysClamp)) {
                                            value_storage[opt_index].u.i64 = value;
                                        }
                                    }
                                    break;
                                case AV_OPT_TYPE_UINT64:
                                    {
                                        uint64_t value = *(uint64_t *)ptr;
                                        uint64_t umin = min;
                                        uint64_t umax = max;

                                        if (!value_storage[opt_index].inited) {
                                            value_storage[opt_index].u.u64 = *(uint64_t *)ptr;
                                            value_storage[opt_index].inited = 1;
                                        }
                                        value = value_storage[opt_index].u.u64;
                                        if (ImGui::DragScalar(opt->name, ImGuiDataType_U64, &value, 1, &umin, &umax, "%lu", ImGuiSliderFlags_AlwaysClamp)) {
                                            value_storage[opt_index].u.u64 = value;
                                        }
                                    }
                                    break;
                                case AV_OPT_TYPE_DOUBLE:
                                    {
                                        double value = *(double *)ptr;

                                        if (!value_storage[opt_index].inited) {
                                            value_storage[opt_index].u.dbl = *(double *)ptr;
                                            value_storage[opt_index].inited = 1;
                                        }
                                        value = value_storage[opt_index].u.dbl;
                                        if (ImGui::DragScalar(opt->name, ImGuiDataType_Double, &value, 1.0, &min, &max, "%f", ImGuiSliderFlags_AlwaysClamp)) {
                                            value_storage[opt_index].u.dbl = value;
                                        }
                                    }
                                    break;
                                case AV_OPT_TYPE_FLOAT:
                                    {
                                        float fmax = max;
                                        float fmin = min;
                                        float value;

                                        if (!value_storage[opt_index].inited) {
                                            value_storage[opt_index].u.flt = *(float *)ptr;
                                            value_storage[opt_index].inited = 1;
                                        }
                                        value = value_storage[opt_index].u.flt;
                                        if (ImGui::DragFloat(opt->name, &value, 1.f, fmin, fmax, "%f", ImGuiSliderFlags_AlwaysClamp))
                                            value_storage[opt_index].u.flt = value;
                                    }
                                    break;
                                case AV_OPT_TYPE_STRING:
                                    {
                                        char string[1024] = { 0 };

                                        if (!value_storage[opt_index].inited) {
                                            av_freep(&value_storage[opt_index].u.str);
                                            value_storage[opt_index].u.str = av_strdup((char *)ptr);
                                            value_storage[opt_index].inited = 1;
                                        }
                                        strncpy(string, value_storage[opt_index].u.str, sizeof(string));
                                        if (ImGui::InputText(opt->name, string, sizeof(string))) {
                                            av_freep(&value_storage[opt_index].u.str);
                                            value_storage[opt_index].u.str = av_strdup(string);
                                        }
                                    }
                                    break;
                                default:
                                    break;
                            }

                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip("%s", opt->help);
                            opt_index++;
                            if (pressed >= 0) {
                                const AVOption *opt = NULL;
                                int idx = 0;
                                while ((opt = av_opt_next(ctx->priv, opt))) {
                                    if (idx == pressed) {
                                        char arg[1024] = { 0 };

                                        switch (opt->type) {
                                            case AV_OPT_TYPE_FLAGS:
                                            case AV_OPT_TYPE_BOOL:
                                            case AV_OPT_TYPE_INT:
                                                snprintf(arg, sizeof(arg) - 1, "%d", value_storage[idx].u.i32);
                                                break;
                                            case AV_OPT_TYPE_INT64:
                                                snprintf(arg, sizeof(arg) - 1, "%ld", value_storage[idx].u.i64);
                                                break;
                                            case AV_OPT_TYPE_UINT64:
                                                snprintf(arg, sizeof(arg) - 1, "%lu", value_storage[idx].u.u64);
                                                break;
                                            case AV_OPT_TYPE_DOUBLE:
                                                snprintf(arg, sizeof(arg) - 1, "%f", value_storage[idx].u.dbl);
                                                break;
                                            case AV_OPT_TYPE_FLOAT:
                                                snprintf(arg, sizeof(arg) - 1, "%f", value_storage[idx].u.flt);
                                                break;
                                            case AV_OPT_TYPE_STRING:
                                                snprintf(arg, FFMIN(sizeof(arg) - 1, strlen(value_storage[idx].u.str)) + 1, "%s", value_storage[idx].u.str);
                                                break;
                                            default:
                                                break;
                                        }

                                        avfilter_graph_send_command(filter_graph, ctx->name, opt->name, arg, NULL, 0, 0);
                                        pressed = -1;
                                        break;
                                    }
                                    idx++;
                                }

                                pressed = -1;
                            }

                            ImGui::PopID();
                        }

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

static void show_dumpgraph(bool *p_open)
{
    if (!graphdump_text || !filter_graph)
        return;

    if (!ImGui::Begin("Graph Dump", p_open, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        return;
    }
    ImGui::Text("%s", graphdump_text);
    ImGui::End();
}

static void show_filters_list(bool *p_open)
{
    static int selected_filter = -2;
    static int prev_selected_filter = -1;
    if (!ImGui::Begin("Filters List", p_open, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        return;
    }
    if (ImGui::BeginListBox("##Filters", ImVec2(400, 300))) {
        static ImGuiTextFilter imgui_filter;
        const AVFilter *filter;
        void *iterator = NULL;
        int n = 0;

        imgui_filter.Draw();
        while ((filter = av_filter_iterate(&iterator))) {
            const bool is_selected = selected_filter == n;
            static bool is_opened = false;

            if (nb_all_filters <= 0) {
                if (!is_source_video_filter(filter)) {
                    continue;
                }
            } else {
                if (!is_simple_video_filter(filter))
                    continue;
            }
            if (!imgui_filter.PassFilter(filter->name))
                continue;

            if (ImGui::Selectable(filter->name, is_selected))
                selected_filter = n;

            if (is_selected)
                ImGui::SetItemDefaultFocus();

            if (ImGui::IsItemActive() || ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", filter->description);
            if (ImGui::IsItemClicked() || ImGui::IsItemActivated()) {
                selected_filter = n;
                is_opened = !is_opened;
            }
            if (is_opened && selected_filter == n) {
                const bool pressed = ImGui::Button("Insert");
                ImGui::SameLine();
                if (prev_selected_filter != selected_filter) {
                    prev_selected_filter = selected_filter;
                    avfilter_graph_free(&probe_graph);
                    if (!probe_graph) {
                        probe_graph = avfilter_graph_alloc();
                        if (!probe_graph)
                            continue;
                        probe_ctx = avfilter_graph_alloc_filter(probe_graph, filter, "probe");
                    }
                    if (!probe_ctx)
                        continue;
                    av_opt_set_defaults(probe_ctx);
                }
                if (pressed) {
                    is_opened = 0;
                    if (probe_ctx) {
                        int ret;
                        filters_options[nb_all_filters].filter_name = av_strdup(probe_ctx->filter->name);
                        if (!filters_options[nb_all_filters].filter_name) {
                            av_log(NULL, AV_LOG_ERROR, "Cannot set filter name\n");
                            break;
                        }

                        filters_options[nb_all_filters].filter_label = av_asprintf("%s%d", probe_ctx->filter->name, nb_all_filters);
                        if (!filters_options[nb_all_filters].filter_label) {
                            av_log(NULL, AV_LOG_ERROR, "Cannot set filter label\n");
                            break;
                        }

                        ret = av_opt_serialize(probe_ctx, 0, AV_OPT_SERIALIZE_SKIP_DEFAULTS,
                                               &filters_options[nb_all_filters].ctx_options, '=', ':');
                        if (ret < 0) {
                            av_freep(&filters_options[nb_all_filters].filter_name);
                            av_log(NULL, AV_LOG_ERROR, "Cannot serialize ctx options\n");
                            break;
                        }

                        ret = av_opt_serialize(probe_ctx->priv, AV_OPT_FLAG_FILTERING_PARAM, AV_OPT_SERIALIZE_SKIP_DEFAULTS,
                                               &filters_options[nb_all_filters].filter_options, '=', ':');
                        if (ret < 0) {
                            av_freep(&filters_options[nb_all_filters].filter_name);
                            av_freep(&filters_options[nb_all_filters].ctx_options);
                            av_log(NULL, AV_LOG_ERROR, "Cannot serialize options\n");
                            break;
                        }
                        nb_all_filters++;
                        need_filters_reinit = 1;
                        avfilter_graph_free(&probe_graph);
                        probe_ctx = NULL;
                    }
                    selected_filter = -2;
                    prev_selected_filter = -1;
                    continue;
                }
                if (filter->priv_class && filter->priv_class->option && probe_ctx) {
                    if (ImGui::TreeNode("Options")) {
                        const AVOption *opt = NULL;
                        int last_offset = -1;
                        double min, max;
                        void *av_class;
                        int index = 0;

                        av_class = probe_ctx->priv;
                        while ((opt = av_opt_next(&filter->priv_class, opt))) {
                            if (last_offset == opt->offset)
                                continue;
                            last_offset = opt->offset;
                            if (!query_ranges((void *)&filter->priv_class, opt, &min, &max))
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
                                        if (str)
                                            memcpy(new_str, str, strlen((const char *)str));
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
                                        AVRational rate = (AVRational){ 0, 0 };
                                        int irate[2] = { 0, 0 };

                                        if (rate.num == 0 && rate.den == 0) {
                                            if (av_opt_get_video_rate(av_class, opt->name, 0, &rate))
                                                av_parse_video_rate(&rate, opt->default_val.str);
                                        }
                                        irate[0] = rate.num;
                                        irate[1] = rate.den;
                                        if (ImGui::DragInt2(opt->name, irate, 1, -8192, 8192)) {
                                            rate.num = irate[0];
                                            rate.den = irate[1];
                                            if (av_opt_set_video_rate(av_class, opt->name, rate, 0))
                                                av_opt_set(av_class, opt->name, av_asprintf("%d/%d", rate.num, rate.den), AV_DICT_DONT_STRDUP_VAL);
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
                                        uint8_t *old_str = NULL;

                                        if (av_opt_get(av_class, opt->name, 0, &old_str))
                                            break;
                                        sscanf((const char *)old_str, "0x%02x%02x%02x%02X", &icol[0], &icol[1], &icol[2], &icol[3]);
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
                                        av_opt_set(av_class, opt->name, av_asprintf("0x%02x%02x%02x%02x", icol[0], icol[1], icol[2], icol[3]), AV_DICT_DONT_STRDUP_VAL);
                                    }
                                    break;
                                case AV_OPT_TYPE_CHLAYOUT:
                                    break;
                                case AV_OPT_TYPE_CONST:
                                    break;
                                default:
                                    break;
                            }

                            if (ImGui::IsItemHovered())
                                ImGui::SetTooltip("%s", opt->help);
                        }
                        ImGui::TreePop();
                    }

                    if (filter->flags & AVFILTER_FLAG_SUPPORT_TIMELINE) {
                        if (ImGui::TreeNode("Timeline")) {
                            char new_str[128] = {0};
                            uint8_t *str = NULL;
                            void *av_class;

                            av_class = probe_ctx;
                            if (av_opt_get(av_class, "enable", 0, &str)) {
                                ImGui::TreePop();
                                continue;
                            }
                            if (str)
                                memcpy(new_str, str, strlen((const char *)str));
                            if (ImGui::InputText("Enable", new_str, IM_ARRAYSIZE(new_str))) {
                                av_opt_set(av_class, "enable", new_str, 0);
                            }
                            ImGui::TreePop();
                        }
                    }
                }
            }
            n++;
        }
        ImGui::EndListBox();
    }
    ImGui::End();
}

int main(int, char**)
{
    GLuint texture;
    int ret;

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
    glGenTextures(1, &texture);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        AVFrame *render_frame;
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        if (show_buffersink_window == false) {
            if (filter_graph) {
                need_filters_reinit = 1;
                if (video_sink_thread.joinable())
                    video_sink_thread.join();
                buffersink_ctx = NULL;
                for (int i = 0; i < nb_all_filters; i++)
                    av_freep(&filters_options[i].filter_name);
                nb_all_filters = 0;
                avfilter_graph_free(&filter_graph);
                need_filters_reinit = 0;
            }
        }

        ret = filters_setup();
        if (ret < 0)
            break;

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        filter_frames_mutex.lock();
        render_frame = filter_frames.empty() ? NULL : filter_frames.back();
        draw_frame(ret, &texture, &show_buffersink_window, render_frame);
        filter_frames_mutex.unlock();
        if (show_filters_list_window)
            show_filters_list(&show_filters_list_window);
        if (show_commands_window)
            show_commands(&show_commands_window);
        if (show_dumpgraph_window)
            show_dumpgraph(&show_dumpgraph_window);

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    need_filters_reinit = 1;
    if (video_sink_thread.joinable())
        video_sink_thread.join();

    av_freep(&graphdump_text);

    for (int i = 0; i < nb_all_filters; i++)
        av_freep(&filters_options[i].filter_name);
    avfilter_graph_free(&filter_graph);
    avfilter_graph_free(&probe_graph);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return ret;
}
