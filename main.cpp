#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <sstream>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imnodes.h"

#include <stdio.h>
#include <unistd.h>

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <AL/alc.h>
#include <AL/al.h>
#include <AL/alext.h>

typedef struct Buffer {
    union {
        ALuint a;
        GLuint v;
    } u;
} Buffer;

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/avstring.h>
#include <libavutil/bprint.h>
#include <libavutil/dict.h>
#include <libavutil/opt.h>
#include <libavutil/intreadwrite.h>
#include <libavutil/imgutils.h>
#include <libavutil/parseutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/time.h>
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include "ringbuffer/ringbuffer.c"
}

static const AVSampleFormat all_sample_fmts[] = {
    AV_SAMPLE_FMT_NONE,
    AV_SAMPLE_FMT_U8, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S32,
    AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_DBL,
    AV_SAMPLE_FMT_U8P, AV_SAMPLE_FMT_S16P, AV_SAMPLE_FMT_S32P,
    AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_DBLP,
    AV_SAMPLE_FMT_S64, AV_SAMPLE_FMT_S64P,
};

#define AL_BUFFERS 16

typedef struct OutputStream {
    AVStream *st;
    AVFilterContext *flt;
    AVCodecContext *enc;

    AVFrame *frame;

    AVPacket *pkt;

    int64_t start_flt_time, end_flt_time, elapsed_flt_time;
    int64_t start_enc_time, end_enc_time, elapsed_enc_time;

    const AVCodec *last_codec;

    uint64_t last_frame_size;
    uint64_t last_packet_size;

    uint64_t sum_of_frames;
    uint64_t sum_of_packets;
} OutputStream;

typedef struct Recorder {
    bool ready;
    char *filename;
    AVFormatContext *format_ctx;
    const AVOutputFormat *oformat;
    std::vector<const AVCodec *> audio_sink_codecs;
    std::vector<const AVCodec *> video_sink_codecs;
    std::vector<OutputStream> ostreams;
} Recorder;

typedef struct FrameInfo {
    int width, height;
    int nb_samples;
    int format;
    int key_frame;
    enum AVPictureType pict_type;
    AVChannelLayout ch_layout;
    AVRational sample_aspect_ratio;
    int64_t pts;
    AVRational time_base;
    int interlaced_frame;
    int top_field_first;
    int sample_rate;
    enum AVColorRange color_range;
    enum AVColorPrimaries color_primaries;
    enum AVColorTransferCharacteristic color_trc;
    enum AVColorSpace colorspace;
    enum AVChromaLocation chroma_location;
    int64_t duration;
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

typedef struct ColorItem {
    float c[4];
} ColorItem;

typedef struct OptStorage {
    union {
        int32_t i32;
        uint32_t u32;
        float flt;
        int64_t i64;
        uint64_t u64;
        double dbl;
        AVRational q;
        char *str;
        ColorItem col;

        int32_t *i32_array;
        uint32_t *u32_array;
        float *flt_array;
        int64_t *i64_array;
        uint64_t *u64_array;
        double *dbl_array;
        AVRational *q_array;
        char **str_array;
        ColorItem *col_array;
    } u;
    unsigned nb_items;
} OptStorage;

typedef struct BufferSource {
    std::string stream_url;
    int stream_index;
    bool ready;
    enum AVMediaType type;
    AVFilterContext *ctx;
    AVFormatContext *fmt_ctx;
    AVCodecContext *dec_ctx;
} BufferSource;

typedef struct BufferSink {
    unsigned id;
    char *label;
    char *description;
    AVFilterContext *ctx;
    AVRational time_base;
    AVRational frame_rate;
    ring_buffer_t empty_frames;
    ring_buffer_t consume_frames;
    ring_buffer_t render_frames;
    double speed;
    bool ready;
    bool fullscreen;
    bool muted;
    bool show_osd;
    bool have_window_pos;
    ImVec2 window_pos;
    ALuint source;
    ALenum format;
    float gain;
    float position[3];
    std::vector<Buffer> buffers;

    GLint downscale_interpolator;
    GLint upscale_interpolator;

    int64_t frame_number;
    int64_t qpts;
    int64_t pts;
    ALint audio_queue_size;
    int sample_rate;
    int frame_nb_samples;

    float *samples;
    unsigned nb_samples;
    unsigned sample_index;

    GLuint texture;
    int width;
    int height;

    FrameInfo frame_info;
} BufferSink;

typedef struct FilterNode {
    unsigned id;
    bool imported_id;
    bool set_pos;
    ImVec2 pos;
    int edge;
    bool colapsed;
    bool have_exports;
    bool show_exports;
    const AVFilter *filter;
    char *filter_name;
    char *filter_label;
    char *ctx_options;
    char *filter_options;
    AVFilterContext *probe;
    AVFilterContext *ctx;
    std::string stream_url;

    std::vector<int> inpad_edges;
    std::vector<int> outpad_edges;

    std::vector<OptStorage> opt_storage;
} FilterNode;

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

bool full_screen = false;
bool show_osd_all = false;
bool muted_all = false;
bool restart_display = false;
int filter_graph_nb_threads = 0;
int filter_graph_auto_convert_flags = 0;
unsigned last_buffersink_window = 0;
unsigned last_abuffersink_window = 0;
unsigned focus_buffersink_window = UINT_MAX;
unsigned focus_abuffersink_window = UINT_MAX;
bool show_abuffersink_window = true;
bool show_buffersink_window = true;
bool show_dumpgraph_window = false;
bool show_commands_window = false;
bool show_filtergraph_editor_window = true;
bool show_mini_map = true;
int mini_map_location = ImNodesMiniMapLocation_BottomRight;
int style_colors = 1;

char *import_script_file_name = NULL;

bool need_filters_reinit = true;
bool need_muxing = false;
std::atomic<bool> framestep = false;
std::atomic<bool> paused = true;
bool show_info = false;
bool show_help = false;
bool show_version = false;
bool show_console = false;
bool show_log_window = false;
bool show_record_window = false;

int log_level = AV_LOG_INFO;
ImGuiTextBuffer log_buffer;
ImVector<int> log_lines_offsets;
ImVector<int> log_lines_levels;
std::mutex log_mutex;

GLint global_upscale_interpolation = GL_NEAREST;
GLint global_downscale_interpolation = GL_NEAREST;

int output_sample_rate = 44100;
int audio_queue_size = AL_BUFFERS;
int display_w;
int display_h;
int width = 1280;
int height = 720;
unsigned depth = 0;
unsigned audio_format = 0;
bool filter_graph_is_valid = false;
AVFilterGraph *filter_graph = NULL;
AVFilterGraph *probe_graph = NULL;
char *graphdump_text = NULL;
float audio_sample_range[2] = { 1.f, 1.f };
float audio_window_size[2] = { 0, 100 };
float osd_fullscreen_pos[2] = { 0.01f, 0.01f };
float osd_alpha = 0.5f;
float commands_alpha = 0.8f;
float console_alpha = 0.5f;
float dump_alpha = 0.7f;
float record_alpha = 0.9f;
float editor_alpha = 1.0f;
float help_alpha = 0.5f;
float info_alpha = 0.7f;
float log_alpha = 0.7f;
float sink_alpha = 1.f;
float version_alpha = 0.8f;

std::vector<Recorder> recorder;
std::vector<std::condition_variable> recorder_cv;
std::vector<std::thread> recorder_threads;
std::vector<std::mutex> recorder_mutexes;

int editor_edge = 0;
ImNodesEditorContext *node_editor_context;

std::mutex filtergraph_mutex;

std::vector<ALuint> play_sources;
std::thread play_sound_thread;

std::vector<BufferSource> buffer_sources;
std::vector<BufferSink> abuffer_sinks;
std::vector<BufferSink> buffer_sinks;
std::vector<std::condition_variable> acv;
std::vector<std::condition_variable> cv;
std::vector<std::mutex> amutexes;
std::vector<std::mutex> mutexes;
std::vector<std::thread> audio_sink_threads;
std::vector<std::thread> video_sink_threads;
std::vector<std::thread> source_threads;
std::vector<FilterNode> filter_nodes;
std::vector<std::pair<int, int>> filter_links;
std::vector<Edge2Pad> edge2pad;

static const GLenum gl_fmts[] = { GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT };
static const enum AVPixelFormat hi_pix_fmts[] = { AV_PIX_FMT_RGBA64, AV_PIX_FMT_NONE };
static const enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_RGBA, AV_PIX_FMT_NONE };
static const enum AVSampleFormat hi_sample_fmts[] = { AV_SAMPLE_FMT_DBLP, AV_SAMPLE_FMT_NONE };
static const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_NONE };
static int sample_rates[] = { output_sample_rate, 0 };

ALCdevice *al_dev = NULL;
ALCcontext *al_ctx = NULL;
float listener_direction[6] = { 0, 0, -1, 0, 1, 0 };
float listener_position[3] = { 0, 0, 0 };

static void alloc_ring_buffer(ring_buffer_t *ring_buffer, Buffer *id)
{
    if (!id)
        return;

    while (!ring_buffer_is_full(ring_buffer)) {
        AVFrame *empty_frame = av_frame_alloc();

        if (empty_frame) {
            unsigned i = ring_buffer_num_items(ring_buffer);
            ring_item_t item = { empty_frame, id[i] };
            ring_buffer_enqueue(ring_buffer, item);
        }
    }
}

static void clear_ring_buffer(ring_buffer_t *ring_buffer)
{
    while (!ring_buffer_is_empty(ring_buffer)) {
        ring_item_t item = { NULL, 0 };

        ring_buffer_dequeue(ring_buffer, &item);
        av_frame_free(&item.frame);
    }
}

static void sound_thread(ALsizei nb_sources, std::vector<ALuint> *sources)
{
    bool step = framestep;
    bool state = paused;

    if ((state == true) || (step == false))
        alSourceStopv(nb_sources, sources->data());

    while (!need_filters_reinit && filter_graph_is_valid) {
        bool new_step = framestep;
        bool new_state = paused;

        if (sources->size() == 0)
            break;

        if ((state != new_state) || (step != new_step)) {
            state = new_state;
            step = new_step;
            if (state == true && step == false)
                alSourcePausev(nb_sources, sources->data());
            else
                alSourcePlayv(nb_sources, sources->data());
        }
        av_usleep(100000);
    }
}

static int write_frame(AVFormatContext *fmt_ctx, OutputStream *os)
{
    AVCodecContext *c = os->enc;
    AVFrame *frame = os->frame;
    AVPacket *pkt = os->pkt;
    AVStream *st = os->st;
    unsigned frame_size = 0;
    int ret;

    switch (c->codec_type) {
    case AVMEDIA_TYPE_AUDIO:
        frame_size = av_samples_get_buffer_size(NULL, frame->ch_layout.nb_channels,
                                                frame->nb_samples, (AVSampleFormat)frame->format, 1);
        break;
    case AVMEDIA_TYPE_VIDEO:
        frame_size = av_image_get_buffer_size((AVPixelFormat)frame->format, frame->width,
                                              frame->height, 1);
        break;
    default:
        break;
    }

    os->last_frame_size = frame_size;
    os->sum_of_frames += frame_size;

    ret = avcodec_send_frame(c, frame);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Error sending a frame to the encoder: %s\n",
               av_err2str(ret));
        return ret;
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(c, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Error encoding a frame: %s\n", av_err2str(ret));
            return ret;
        }

        os->last_packet_size = pkt->size;
        os->sum_of_packets += pkt->size;

        av_packet_rescale_ts(pkt, c->time_base, st->time_base);
        pkt->stream_index = st->index;

        ret = av_interleaved_write_frame(fmt_ctx, pkt);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Error while writing output packet: %s\n", av_err2str(ret));
            return ret;
        }
    }

    return ret == AVERROR_EOF ? AVERROR_EOF : 0;
}

static void kill_recorder_threads()
{
    for (unsigned i = 0; i < recorder_threads.size(); i++) {
        recorder[i].ready = false;
        if (recorder_threads[i].joinable()) {
            recorder_threads[i].join();
        }
    }
}

static void recorder_thread(Recorder *recorder, std::mutex *mutex, std::condition_variable *cv)
{
    while (recorder->format_ctx) {
        AVFormatContext *format_ctx = recorder->format_ctx;
        unsigned out_stream_eof = 0;

        if (filter_graph_is_valid == false)
            break;

        for (unsigned i = 0; i < recorder->ostreams.size(); i++) {
            OutputStream *os = &recorder->ostreams[i];
            int ret;

            if (!os->flt)
                break;

            os->start_flt_time = av_gettime_relative();

            ret = av_buffersink_get_frame_flags(os->flt, os->frame, 0);

            os->end_flt_time = av_gettime_relative();
            os->elapsed_flt_time += os->end_flt_time - os->start_flt_time;

            if (ret == AVERROR_EOF)
                out_stream_eof++;

            if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN))
                break;

            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                continue;

            os->start_enc_time = av_gettime_relative();

            ret = write_frame(format_ctx, os);
            if (ret == AVERROR_EOF)
                out_stream_eof++;

            os->end_enc_time = av_gettime_relative();
            os->elapsed_enc_time += os->end_enc_time - os->start_enc_time;

            av_frame_unref(os->frame);
            if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN))
                break;
        }

        if (out_stream_eof == recorder->ostreams.size())
            break;
    }

    if (recorder->format_ctx) {
        AVFormatContext *format_ctx = recorder->format_ctx;
        av_write_trailer(format_ctx);
        avio_closep(&format_ctx->pb);

        for (unsigned i = 0; i < recorder->ostreams.size(); i++) {
            OutputStream *os = &recorder->ostreams[i];

            os->flt = NULL;
            av_frame_free(&os->frame);
            av_packet_free(&os->pkt);
            avcodec_free_context(&os->enc);
        }

        avformat_free_context(format_ctx);
        recorder->format_ctx = NULL;
        av_freep(&recorder->filename);
    }
}

static void worker_thread(BufferSink *sink, std::mutex *mutex, std::condition_variable *cv)
{
    int ret = 0;

    while (sink->ctx) {
        if (need_filters_reinit == true)
            break;
        if (filter_graph_is_valid == false)
            break;

        std::unique_lock<std::mutex> lk(*mutex);
        cv->wait(lk, [sink]{ return sink->ready; });
        if (sink->ready == false)
            continue;
        sink->ready = false;

        while (!ring_buffer_is_empty(&sink->empty_frames)) {
            ring_item_t item = { NULL, 0 };
            int64_t start, end;

            if (need_filters_reinit == true)
                break;
            if (filter_graph_is_valid == false)
                break;

            if (ring_buffer_is_full(&sink->consume_frames))
                break;

            ring_buffer_dequeue(&sink->empty_frames, &item);
            if (!item.frame)
                break;

            filtergraph_mutex.lock();
            start = av_gettime_relative();
            ret = av_buffersink_get_frame_flags(sink->ctx, item.frame, 0);
            end = av_gettime_relative();
            filtergraph_mutex.unlock();
            if (end > start && item.frame)
                sink->speed = 1000000. * (std::max(item.frame->nb_samples, 1)) * av_q2d(av_inv_q(sink->frame_rate)) / (end - start);
            if (ret < 0) {
                av_frame_unref(item.frame);
                ring_buffer_enqueue(&sink->empty_frames, item);
                if (ret != AVERROR(EAGAIN))
                    break;
                item.frame = NULL;
            }

            if (item.frame)
                ring_buffer_enqueue(&sink->consume_frames, item);
        }

        if (ret < 0 && ret != AVERROR(EAGAIN))
            break;
    }

    clear_ring_buffer(&sink->empty_frames);
    clear_ring_buffer(&sink->consume_frames);
    clear_ring_buffer(&sink->render_frames);
}

static void notify_worker(BufferSink *sink, std::mutex *mutex, std::condition_variable *cv)
{
    sink->ready = true;
    cv->notify_one();
}

static void kill_audio_sink_threads()
{
    for (unsigned i = 0; i < audio_sink_threads.size(); i++) {
        BufferSink *sink = &abuffer_sinks[i];

        if (audio_sink_threads[i].joinable()) {
            notify_worker(sink, &amutexes[i], &acv[i]);
            audio_sink_threads[i].join();
        }

        av_freep(&sink->label);
        av_freep(&sink->description);
        av_freep(&sink->samples);
        alDeleteSources(1, &sink->source);
        for (unsigned n = 0; n < sink->buffers.size(); n++)
            alDeleteBuffers(1, &sink->buffers[n].u.a);

        ring_buffer_free(&sink->empty_frames);
        ring_buffer_free(&sink->consume_frames);
        ring_buffer_free(&sink->render_frames);

        sink->buffers.clear();
    }
}

static void kill_video_sink_threads()
{
    for (unsigned i = 0; i < video_sink_threads.size(); i++) {
        BufferSink *sink = &buffer_sinks[i];

        if (video_sink_threads[i].joinable()) {
            notify_worker(sink, &mutexes[i], &cv[i]);
            video_sink_threads[i].join();
        }

        av_freep(&sink->label);
        av_freep(&sink->description);
        glDeleteTextures(1, &sink->buffers[0].u.v);

        ring_buffer_free(&sink->empty_frames);
        ring_buffer_free(&sink->consume_frames);
        ring_buffer_free(&sink->render_frames);

        sink->buffers.clear();
    }
}

static void kill_source_threads()
{
    for (unsigned i = 0; i < source_threads.size(); i++) {
        buffer_sources[i].ready = false;
        if (source_threads[i].joinable()) {
            source_threads[i].join();
        }
    }
}

static int get_nb_filter_threads(const AVFilter *filter)
{
    if (filter->flags & AVFILTER_FLAG_SLICE_THREADS)
        return 0;
    return 1;
}

static void source_worker_thread(BufferSource *source)
{
    const int stream_index = source->stream_index;
    AVFilterContext *buffersrc_ctx = source->ctx;
    AVFormatContext *fmt_ctx = source->fmt_ctx;
    AVCodecContext *dec_ctx = source->dec_ctx;
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    int ret;

    while (source->ready == true) {
        if (need_filters_reinit == true)
            break;
        if (filter_graph_is_valid == false)
            break;

        if (!av_buffersrc_get_nb_failed_requests(buffersrc_ctx)) {
            av_usleep(1000);
            continue;
        }

        if ((ret = av_read_frame(fmt_ctx, packet)) < 0)
            break;

        if (packet->stream_index == stream_index) {
            ret = avcodec_send_packet(dec_ctx, packet);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Error while sending a packet to the decoder\n");
                break;
            }

            while (ret >= 0) {
                if (source->ready == false)
                    break;
                if (need_filters_reinit == true)
                    break;
                if (filter_graph_is_valid == false)
                    break;
                ret = avcodec_receive_frame(dec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    av_log(NULL, AV_LOG_ERROR, "Error while receiving a frame from the decoder\n");
                    break;
                }

                if (ret >= 0) {
                    if (av_buffersrc_add_frame_flags(buffersrc_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF) < 0) {
                        av_log(NULL, AV_LOG_ERROR, "Error while feeding the filtergraph\n");
                        break;
                    }
                }
            }
        }
        av_packet_unref(packet);
    }

    if (ret == AVERROR_EOF) {
        /* signal EOF to the filtergraph */
        if (av_buffersrc_add_frame_flags(buffersrc_ctx, NULL, 0) < 0) {
            av_log(NULL, AV_LOG_ERROR, "Error while closing the filter source\n");
        }
    }

    av_packet_free(&packet);
    av_frame_free(&frame);

    avcodec_free_context(&source->dec_ctx);
    avformat_close_input(&source->fmt_ctx);
}

static void find_source_params(BufferSource *source)
{
    AVFilterContext *buffersrc_ctx = source->ctx;
    int stream_index, ret;
    const AVCodec *dec;

    if (source->stream_url.empty())
        return;

    if ((ret = avformat_open_input(&source->fmt_ctx, source->stream_url.c_str(), NULL, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open input\n");
        return;
    }

    if ((ret = avformat_find_stream_info(source->fmt_ctx, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot find stream information\n");
        return;
    }

    /* select the audio stream */
    ret = av_find_best_stream(source->fmt_ctx, source->type, -1, -1, &dec, 0);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot find an stream in the input file\n");
        return;
    }
    stream_index = source->stream_index = ret;

    /* create decoding context */
    source->dec_ctx = avcodec_alloc_context3(dec);
    if (!source->dec_ctx)
        return;
    avcodec_parameters_to_context(source->dec_ctx, source->fmt_ctx->streams[stream_index]->codecpar);

    if ((ret = avcodec_open2(source->dec_ctx, dec, NULL)) < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot open decoder\n");
        return;
    }

    AVBufferSrcParameters *params = av_buffersrc_parameters_alloc();
    params->time_base = source->fmt_ctx->streams[stream_index]->time_base;
    switch (source->type) {
    case AVMEDIA_TYPE_AUDIO:
        params->format = source->dec_ctx->sample_fmt;
        params->sample_rate = source->dec_ctx->sample_rate;
        av_channel_layout_copy(&params->ch_layout, &source->dec_ctx->ch_layout);
        break;
    case AVMEDIA_TYPE_VIDEO:
        params->format = source->dec_ctx->pix_fmt;
        params->frame_rate = source->dec_ctx->framerate;
        params->width = source->dec_ctx->width;
        params->height = source->dec_ctx->height;
        params->color_range = source->dec_ctx->color_range;
        params->color_space = source->dec_ctx->colorspace;
        break;
    default:
        break;
    }
    av_buffersrc_parameters_set(buffersrc_ctx, params);
    av_free(params);
    source->ready = true;
}

static int filters_setup()
{
    const AVFilter *new_filter;
    int ret;

    if (need_filters_reinit == false)
        return 0;

    kill_recorder_threads();
    recorder_cv.clear();
    recorder_mutexes.clear();

    if (play_sound_thread.joinable())
        play_sound_thread.join();
    play_sources.clear();

    kill_source_threads();
    kill_audio_sink_threads();
    kill_video_sink_threads();

    source_threads.clear();
    audio_sink_threads.clear();
    video_sink_threads.clear();

    need_filters_reinit = false;
    filter_graph_is_valid = false;

    if (filter_nodes.size() == 0)
        return 0;

    buffer_sources.clear();
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

        filter_nodes[i].ctx = filter_ctx;

        ret = av_opt_copy(filter_ctx, filter_nodes[i].probe);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Cannot copy options for filter.\n");
            goto error;
        }

        ret = av_opt_copy(filter_ctx->priv, filter_nodes[i].probe->priv);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Cannot copy private options for filter.\n");
            goto error;
        }

        if (!strcmp(filter_ctx->filter->name, "buffer")) {
            BufferSource new_source;

            new_source.ready = false;
            new_source.fmt_ctx = NULL;
            new_source.dec_ctx = NULL;
            new_source.ctx = filter_ctx;
            if (!filter_nodes[i].stream_url.empty())
                new_source.stream_url = filter_nodes[i].stream_url;
            new_source.type = AVMEDIA_TYPE_VIDEO;
            find_source_params(&new_source);
            buffer_sources.push_back(new_source);
        } else if (!strcmp(filter_ctx->filter->name, "abuffer")) {
            BufferSource new_source;

            new_source.ready = false;
            new_source.fmt_ctx = NULL;
            new_source.dec_ctx = NULL;
            new_source.ctx = filter_ctx;
            if (!filter_nodes[i].stream_url.empty())
                new_source.stream_url = filter_nodes[i].stream_url;
            new_source.type = AVMEDIA_TYPE_AUDIO;
            find_source_params(&new_source);
            buffer_sources.push_back(new_source);
        } else if (!strcmp(filter_ctx->filter->name, "buffersink")) {
            const AVPixelFormat *encoder_fmts = (recorder.size() > 0 && recorder[0].video_sink_codecs.size() > 0) ? recorder[0].video_sink_codecs[buffer_sinks.size()]->pix_fmts : NULL;
            BufferSink new_sink = {0};

            new_sink.ctx = filter_ctx;
            new_sink.ready = false;
            new_sink.have_window_pos = false;
            new_sink.fullscreen = false;
            new_sink.muted = false;
            new_sink.show_osd = true;
            new_sink.frame_number = 0;
            new_sink.upscale_interpolator = global_upscale_interpolation;
            new_sink.downscale_interpolator = global_downscale_interpolation;
            ret = av_opt_set_int_list(filter_ctx, "pix_fmts", need_muxing ? encoder_fmts : depth ? hi_pix_fmts : pix_fmts,
                                      AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
            if (ret < 0) {
                ret = av_opt_set_array(filter_ctx, "pixel_formats", AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE, 0, 1,
                                       AV_OPT_TYPE_PIXEL_FMT, need_muxing ? encoder_fmts : depth ? hi_pix_fmts : pix_fmts);
                if (ret < 0) {
                    av_log(NULL, AV_LOG_ERROR, "Cannot set buffersink output pixel format.\n");
                    goto error;
                }
            }

            buffer_sinks.push_back(new_sink);
        } else if (!strcmp(filter_ctx->filter->name, "abuffersink")) {
            const AVSampleFormat *encoder_fmts = (recorder.size() > 0 && recorder[0].audio_sink_codecs.size() > 0) ? recorder[0].audio_sink_codecs[abuffer_sinks.size()]->sample_fmts : NULL;
            const int *encoder_samplerates = (recorder.size() > 0 && recorder[0].audio_sink_codecs.size() > 0) ? recorder[0].audio_sink_codecs[abuffer_sinks.size()]->supported_samplerates : NULL;
            const int *encode_samplerates = encoder_samplerates ? encoder_samplerates : sample_rates;
            BufferSink new_sink = {0};

            alcGetIntegerv(al_dev, ALC_FREQUENCY, 1, &new_sink.sample_rate);
            sample_rates[0] = new_sink.sample_rate;

            new_sink.ctx = filter_ctx;
            new_sink.ready = false;
            new_sink.have_window_pos = false;
            new_sink.fullscreen = false;
            new_sink.muted = false;
            new_sink.show_osd = true;
            new_sink.upscale_interpolator = 0;
            new_sink.downscale_interpolator = 0;
            new_sink.frame_number = 0;
            ret = av_opt_set_int_list(filter_ctx, "sample_fmts", need_muxing ? encoder_fmts : audio_format ? hi_sample_fmts : sample_fmts,
                                      AV_SAMPLE_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
            if (ret < 0) {
                ret = av_opt_set_array(filter_ctx, "sample_formats", AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE, 0, 1,
                                       AV_OPT_TYPE_SAMPLE_FMT, need_muxing ? encoder_fmts : audio_format ? hi_sample_fmts : sample_fmts);
                if (ret < 0) {
                    av_log(NULL, AV_LOG_ERROR, "Cannot set abuffersink output sample formats.\n");
                    goto error;
                }
            }

            ret = av_opt_set_int_list(filter_ctx, "sample_rates", sample_rates,
                                      0, AV_OPT_SEARCH_CHILDREN);
            if (ret < 0) {
                ret = av_opt_set_array(filter_ctx, "sample_rates", AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE, 0, 1,
                                       AV_OPT_TYPE_INT, need_muxing ? encode_samplerates : sample_rates);
                if (ret < 0) {
                    av_log(NULL, AV_LOG_ERROR, "Cannot set abuffersink output sample rates.\n");
                    goto error;
                }
            }

            abuffer_sinks.push_back(new_sink);
        }

        ret = avfilter_init_str(filter_ctx, NULL);
        if (ret < 0) {
            av_log(NULL, AV_LOG_ERROR, "Cannot init filter.\n");
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

    if (need_muxing) {
        show_abuffersink_window = false;
        show_buffersink_window = false;

        if (recorder.size() > 0) {
            avformat_alloc_output_context2(&recorder[0].format_ctx, recorder[0].oformat, NULL, recorder[0].filename);
            if (!recorder[0].format_ctx) {
                av_log(NULL, AV_LOG_ERROR, "Could not create output format context.\n");
                ret = AVERROR(ENOMEM);
                goto error;
            }

            recorder[0].ostreams.resize(recorder[0].audio_sink_codecs.size() + recorder[0].video_sink_codecs.size());

            for (unsigned i = 0; i < recorder[0].audio_sink_codecs.size(); i++) {
                BufferSink *sink = &abuffer_sinks[i];
                AVChannelLayout ch_layout;
                const AVCodec *codec;

                codec = recorder[0].audio_sink_codecs[i];
                if (!codec) {
                    av_log(NULL, AV_LOG_ERROR, "No encoder set for %u audio stream\n", i);
                    ret = AVERROR(EINVAL);
                    goto error;
                }

                recorder[0].ostreams[i].last_codec = codec;
                recorder[0].ostreams[i].elapsed_enc_time = 0;
                recorder[0].ostreams[i].start_enc_time = 0;
                recorder[0].ostreams[i].end_enc_time = 0;
                recorder[0].ostreams[i].elapsed_flt_time = 0;
                recorder[0].ostreams[i].start_flt_time = 0;
                recorder[0].ostreams[i].end_flt_time = 0;
                recorder[0].ostreams[i].last_frame_size = 0;
                recorder[0].ostreams[i].last_packet_size = 0;
                recorder[0].ostreams[i].sum_of_frames = 0;
                recorder[0].ostreams[i].sum_of_packets = 0;
                recorder[0].ostreams[i].flt = sink->ctx;
                recorder[0].ostreams[i].enc = avcodec_alloc_context3(codec);
                if (!recorder[0].ostreams[i].enc) {
                    av_log(NULL, AV_LOG_ERROR, "Could not allocate context for %u audio stream\n", i);
                    ret = AVERROR(ENOMEM);
                    goto error;
                }

                recorder[0].ostreams[i].st = avformat_new_stream(recorder[0].format_ctx, NULL);
                if (!recorder[0].ostreams[i].st) {
                    av_log(NULL, AV_LOG_ERROR, "Could not allocate audio stream %u\n", i);
                    goto error;
                }
                recorder[0].ostreams[i].st->id = recorder[0].format_ctx->nb_streams-1;

                recorder[0].ostreams[i].enc->sample_fmt = (AVSampleFormat)av_buffersink_get_format(sink->ctx);
                recorder[0].ostreams[i].enc->time_base = av_buffersink_get_time_base(sink->ctx);
                recorder[0].ostreams[i].enc->sample_rate = av_buffersink_get_sample_rate(sink->ctx);
                av_buffersink_get_ch_layout(sink->ctx, &ch_layout);
                av_channel_layout_copy(&recorder[0].ostreams[i].enc->ch_layout, &ch_layout);
                recorder[0].ostreams[i].st->time_base = recorder[0].ostreams[i].enc->time_base;

                if (recorder[0].oformat->flags & AVFMT_GLOBALHEADER)
                    recorder[0].ostreams[i].enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

                ret = avcodec_open2(recorder[0].ostreams[i].enc, codec, NULL);
                if (ret < 0) {
                    av_log(NULL, AV_LOG_ERROR, "Error opening audio encoder for stream %u\n", i);
                    goto error;
                }

                ret = avcodec_parameters_from_context(recorder[0].ostreams[i].st->codecpar, recorder[0].ostreams[i].enc);
                if (ret < 0) {
                    av_log(NULL, AV_LOG_ERROR, "Error in codec parameters for audio stream %u\n", i);
                    goto error;
                }

                recorder[0].ostreams[i].pkt = av_packet_alloc();
                if (!recorder[0].ostreams[i].pkt) {
                    ret = AVERROR(ENOMEM);
                    av_log(NULL, AV_LOG_ERROR, "Error to allocate packet for audio stream %u\n", i);
                    goto error;
                }

                recorder[0].ostreams[i].frame = av_frame_alloc();
                if (!recorder[0].ostreams[i].frame) {
                    ret = AVERROR(ENOMEM);
                    av_log(NULL, AV_LOG_ERROR, "Error to allocate frame for audio stream %u\n", i);
                    goto error;
                }
            }

            for (unsigned i = 0; i < recorder[0].video_sink_codecs.size(); i++) {
                const unsigned oi = recorder[0].audio_sink_codecs.size() + i;
                BufferSink *sink = &buffer_sinks[i];
                const AVCodec *codec;

                codec = recorder[0].video_sink_codecs[i];
                if (!codec) {
                    av_log(NULL, AV_LOG_ERROR, "No encoder set for %u video stream\n", i);
                    ret = AVERROR(EINVAL);
                    goto error;
                }

                recorder[0].ostreams[oi].last_codec = codec;
                recorder[0].ostreams[oi].elapsed_enc_time = 0;
                recorder[0].ostreams[oi].start_enc_time = 0;
                recorder[0].ostreams[oi].end_enc_time = 0;
                recorder[0].ostreams[oi].elapsed_flt_time = 0;
                recorder[0].ostreams[oi].start_flt_time = 0;
                recorder[0].ostreams[oi].end_flt_time = 0;
                recorder[0].ostreams[oi].last_frame_size = 0;
                recorder[0].ostreams[oi].last_packet_size = 0;
                recorder[0].ostreams[oi].sum_of_frames = 0;
                recorder[0].ostreams[oi].sum_of_packets = 0;
                recorder[0].ostreams[oi].flt = sink->ctx;
                recorder[0].ostreams[oi].enc = avcodec_alloc_context3(codec);
                if (!recorder[0].ostreams[oi].enc) {
                    av_log(NULL, AV_LOG_ERROR, "Could not allocate context for %u video stream\n", i);
                    ret = AVERROR(ENOMEM);
                    goto error;
                }

                recorder[0].ostreams[oi].st = avformat_new_stream(recorder[0].format_ctx, NULL);
                if (!recorder[0].ostreams[oi].st) {
                    av_log(NULL, AV_LOG_ERROR, "Could not allocate video stream %u\n", i);
                    goto error;
                }
                recorder[0].ostreams[oi].st->id = recorder[0].format_ctx->nb_streams-1;

                recorder[0].ostreams[oi].enc->width = (AVPixelFormat)av_buffersink_get_w(sink->ctx);
                recorder[0].ostreams[oi].enc->height = (AVPixelFormat)av_buffersink_get_h(sink->ctx);
                recorder[0].ostreams[oi].enc->pix_fmt = (AVPixelFormat)av_buffersink_get_format(sink->ctx);
                recorder[0].ostreams[oi].enc->time_base = av_buffersink_get_time_base(sink->ctx);
                recorder[0].ostreams[oi].st->time_base = recorder[0].ostreams[oi].enc->time_base;

                if (recorder[0].oformat->flags & AVFMT_GLOBALHEADER)
                    recorder[0].ostreams[oi].enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

                ret = avcodec_open2(recorder[0].ostreams[oi].enc, codec, NULL);
                if (ret < 0) {
                    av_log(NULL, AV_LOG_ERROR, "Error opening video encoder for stream %u\n", i);
                    goto error;
                }

                ret = avcodec_parameters_from_context(recorder[0].ostreams[oi].st->codecpar, recorder[0].ostreams[oi].enc);
                if (ret < 0) {
                    av_log(NULL, AV_LOG_ERROR, "Error in codec parameters for video stream %u\n", i);
                    goto error;
                }

                recorder[0].ostreams[oi].pkt = av_packet_alloc();
                if (!recorder[0].ostreams[oi].pkt) {
                    ret = AVERROR(ENOMEM);
                    av_log(NULL, AV_LOG_ERROR, "Error to allocate packet for video stream %u\n", i);
                    goto error;
                }

                recorder[0].ostreams[oi].frame = av_frame_alloc();
                if (!recorder[0].ostreams[oi].frame) {
                    ret = AVERROR(ENOMEM);
                    av_log(NULL, AV_LOG_ERROR, "Error to allocate frame for video stream %u\n", i);
                    goto error;
                }
            }

            ret = avio_open(&recorder[0].format_ctx->pb, recorder[0].filename, AVIO_FLAG_WRITE);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Could not open '%s': %s\n", recorder[0].filename, av_err2str(ret));
                goto error;
            }

            ret = avformat_write_header(recorder[0].format_ctx, NULL);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Error occurred when writing header: %s\n", av_err2str(ret));
                goto error;
            }

            std::vector<std::condition_variable> recorder_cv_list(recorder.size());
            recorder_cv.swap(recorder_cv_list);

            std::vector<std::mutex> recorder_mutex_list(recorder.size());
            recorder_mutexes.swap(recorder_mutex_list);

            std::vector<std::thread> recorder_thread_list(recorder.size());
            recorder_threads.swap(recorder_thread_list);

            for (unsigned i = 0; i < recorder.size(); i++) {
                std::thread rec_thread(recorder_thread, &recorder[i], &recorder_mutexes[i], &recorder_cv[i]);

                recorder_threads[i].swap(rec_thread);
            }
        }
    } else {
        show_abuffersink_window = true;
        show_buffersink_window = true;
    }

error:

    need_muxing = false;

    if (ret < 0) {
        if (recorder.size() > 0) {
            avformat_free_context(recorder[0].format_ctx);
            recorder[0].format_ctx = NULL;
        }
        return ret;
    }

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
        sink->description = NULL;
        sink->time_base = av_buffersink_get_time_base(sink->ctx);
        sink->frame_rate = av_buffersink_get_frame_rate(sink->ctx);
        sink->pts = AV_NOPTS_VALUE;
        sink->sample_index = 0;
        sink->samples = NULL;
        sink->frame_number = 0;
        sink->frame_nb_samples = 0;
        sink->nb_samples = 0;
        sink->texture = 0;
        sink->width = 0;
        sink->height = 0;
        sink->buffers.resize(4);
        ring_buffer_init(&sink->empty_frames, sink->buffers.size());
        ring_buffer_init(&sink->consume_frames, sink->buffers.size());
        ring_buffer_init(&sink->render_frames, sink->buffers.size());

        glGenTextures(1, &sink->buffers[0].u.v);
        for (unsigned n = 1; n < sink->buffers.size(); n++)
            sink->buffers[n].u.v = sink->buffers[0].u.v;

        alloc_ring_buffer(&sink->empty_frames, sink->buffers.data());
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
        char layout_name[512] = { 0 };
        BufferSink *sink = &abuffer_sinks[i];
        AVChannelLayout layout;

        av_buffersink_get_ch_layout(sink->ctx, &layout);
        av_channel_layout_describe(&layout, layout_name, sizeof(layout_name));

        sink->id = i;
        sink->label = av_asprintf("Audio FilterGraph Output %d", i);
        sink->description = av_asprintf("%s", layout_name);
        sink->time_base = av_buffersink_get_time_base(sink->ctx);
        sink->frame_rate = av_make_q(av_buffersink_get_sample_rate(sink->ctx), 1);
        sink->sample_index = 0;
        sink->nb_samples = 512;
        sink->frame_number = 0;
        sink->frame_nb_samples = 0;
        sink->pts = AV_NOPTS_VALUE;
        sink->samples = (float *)av_calloc(sink->nb_samples, sizeof(float));
        sink->audio_queue_size = audio_queue_size;
        sink->texture = 0;
        sink->width = 0;
        sink->height = 0;
        sink->buffers.resize(sink->audio_queue_size);
        ring_buffer_init(&sink->empty_frames, audio_queue_size);
        ring_buffer_init(&sink->consume_frames, audio_queue_size);
        ring_buffer_init(&sink->render_frames, audio_queue_size);

        sink->format = audio_format ? AL_FORMAT_MONO_DOUBLE_EXT : AL_FORMAT_MONO_FLOAT32;

        for (unsigned n = 0; n < sink->buffers.size(); n++)
            alGenBuffers(1, &sink->buffers[n].u.a);

        alGenSources(1, &sink->source);
        play_sources.push_back(sink->source);
        sink->gain = 1.f;
        sink->position[0] =  0.f;
        sink->position[1] =  0.f;
        sink->position[2] = -1.f;
        alSource3f(sink->source, AL_POSITION, sink->position[0], sink->position[1], sink->position[2]);
        alSourcei(sink->source, AL_SOURCE_RELATIVE, AL_TRUE);
        alSourcei(sink->source, AL_ROLLOFF_FACTOR, 0);

        alloc_ring_buffer(&sink->empty_frames, sink->buffers.data());
        std::thread asink_thread(worker_thread, &abuffer_sinks[i], &amutexes[i], &acv[i]);

        audio_sink_threads[i].swap(asink_thread);
    }

    std::thread new_sound_thread(sound_thread, abuffer_sinks.size(), &play_sources);
    play_sound_thread.swap(new_sound_thread);

    std::vector<std::thread> bufferthread_list(buffer_sources.size());
    source_threads.swap(bufferthread_list);

    for (unsigned i = 0; i < buffer_sources.size(); i++) {
        std::thread source_thread(source_worker_thread, &buffer_sources[i]);

        source_threads[i].swap(source_thread);
    }

    return 0;
}

static void load_frame(GLuint *out_texture, int *width, int *height, AVFrame *frame,
                       BufferSink *sink)
{
    const unsigned idx = (frame->format == AV_PIX_FMT_RGBA) ? 0 : 1;
    const size_t pixel_size = idx ? 4 * sizeof(uint16_t) : 4 * sizeof(uint8_t);

    *width  = frame->width;
    *height = frame->height;

    glBindTexture(GL_TEXTURE_2D, *out_texture);
    if (sink->pts != frame->pts) {
        sink->texture = *out_texture;
        sink->width = *width;
        sink->height = *height;
        sink->pts = frame->pts;
        sink->frame_number++;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, sink->downscale_interpolator);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, sink->upscale_interpolator);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, frame->linesize[0] / pixel_size);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frame->width, frame->height, 0, GL_RGBA, gl_fmts[idx], frame->data[0]);
    }
}

static void draw_info(bool *p_open, bool full)
{
    BufferSink *last_sink = NULL;
    FrameInfo *frame = NULL;
    int nb_columns = 0;
    const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration |
                                          ImGuiWindowFlags_AlwaysAutoResize |
                                          ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoNav |
                                          ImGuiWindowFlags_NoFocusOnAppearing |
                                          ImGuiWindowFlags_HorizontalScrollbar |
                                          ImGuiWindowFlags_NoMove;

    if (full == false) {
        if (last_buffersink_window >= 0 && last_buffersink_window < buffer_sinks.size()) {
            last_sink = &buffer_sinks[last_buffersink_window];
            frame = &last_sink->frame_info;
        }

        if (last_abuffersink_window >= 0 && last_abuffersink_window < abuffer_sinks.size()) {
            last_sink = &abuffer_sinks[last_abuffersink_window];
            frame = &last_sink->frame_info;
        }

        if (!frame)
            return;
    } else {
        if (!(last_buffersink_window >= 0 && last_buffersink_window < buffer_sinks.size()) &&
            !(last_abuffersink_window >= 0 && last_abuffersink_window < abuffer_sinks.size()))
            return;
    }

    ImGui::SetNextWindowPos(ImVec2(display_w/2, display_h/2), 0, ImVec2(0.5, 0.5));
    ImGui::SetNextWindowBgAlpha(info_alpha);
    ImGui::SetNextWindowFocus();

    if (!ImGui::Begin("##Info", p_open, window_flags)) {
        ImGui::End();
        return;
    }

    if (full) {
        nb_columns = ceilf(sqrtf(buffer_sinks.size()+abuffer_sinks.size()));
        ImGui::BeginTable("###FullInfo", nb_columns, ImGuiTableFlags_Resizable |
                                                     ImGuiTableFlags_NoSavedSettings |
                                                     ImGuiTableFlags_ScrollX |
                                                     ImGuiTableFlags_ScrollY);
    }

    for (size_t i = 0; i < (full ? (buffer_sinks.size()+abuffer_sinks.size()) : 1); i++) {
        BufferSink *sink = NULL;

        if (full) {
            if (i && (i % nb_columns) == 0)
                ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (i >= buffer_sinks.size() && abuffer_sinks.size() > 0) {
                sink = &abuffer_sinks[i-buffer_sinks.size()];
                frame = &sink->frame_info;
                if (!sink->label)
                    continue;
                ImGui::Spacing();
                ImGui::TextUnformatted(sink->label);
            } else {
                sink = &buffer_sinks[i];
                frame = &sink->frame_info;
                if (!sink->label)
                    continue;
                ImGui::Spacing();
                ImGui::TextUnformatted(sink->label);
            }
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
        } else if (frame->sample_rate > 0) {
            char chlayout_name[1024] = {0};

            ImGui::Text("SAMPLES: %d", frame->nb_samples);
            ImGui::Separator();
            ImGui::Text("KEY FRAME: %d", frame->key_frame);
            ImGui::Separator();
            ImGui::Text("SAMPLE RATE: %d", frame->sample_rate);
            ImGui::Separator();
            ImGui::Text("SAMPLE FORMAT: %s", av_get_sample_fmt_name((enum AVSampleFormat)frame->format));
            ImGui::Separator();
            av_channel_layout_describe(&frame->ch_layout, chlayout_name, sizeof(chlayout_name));
            ImGui::Text("CHANNEL LAYOUT: %s", chlayout_name);
            ImGui::Separator();
        }
        ImGui::Text("PTS: %ld", frame->pts);
        ImGui::Separator();
        ImGui::Text("DURATION: %ld", frame->duration);
        ImGui::Separator();
        if (frame->time_base.num && frame->time_base.den)
            ImGui::Text("TIME BASE: %d/%d", frame->time_base.num, frame->time_base.den);
        else if (last_sink)
            ImGui::Text("TIME BASE: %d/%d", last_sink->time_base.num, last_sink->time_base.den);
        else if (sink)
            ImGui::Text("TIME BASE: %d/%d", sink->time_base.num, sink->time_base.den);
        if (full && sink) {
            ImGui::Separator();
            ImGui::Text("TIME:  %.5f", frame->pts != AV_NOPTS_VALUE ? av_q2d(sink->time_base) * frame->pts : NAN);
        }
    }

    if (full)
        ImGui::EndTable();

    ImGui::End();
}

static void draw_version(bool *p_open)
{
    const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration |
                                          ImGuiWindowFlags_AlwaysAutoResize |
                                          ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoNav |
                                          ImGuiWindowFlags_HorizontalScrollbar |
                                          ImGuiWindowFlags_NoFocusOnAppearing |
                                          ImGuiWindowFlags_NoMove;

    ImGui::SetNextWindowPos(ImVec2(display_w/2, display_h/2), 0, ImVec2(0.5, 0.5));
    ImGui::SetNextWindowSize(ImVec2(display_w, display_h));
    ImGui::SetNextWindowBgAlpha(version_alpha);
    ImGui::SetNextWindowFocus();

    if (!ImGui::Begin("##Version", p_open, window_flags)) {
        ImGui::End();
        return;
    }

    ImGui::Text("libavutil: %s", LIBAVUTIL_IDENT);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(avutil_configuration());
    ImGui::PopTextWrapPos();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(avutil_license());
    ImGui::PopTextWrapPos();
    ImGui::Separator();
    ImGui::Text("libavfilter: %s", LIBAVFILTER_IDENT);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(avfilter_configuration());
    ImGui::PopTextWrapPos();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(avfilter_license());
    ImGui::PopTextWrapPos();
    ImGui::Separator();
    ImGui::Text("libavcodec: %s", LIBAVCODEC_IDENT);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(avcodec_configuration());
    ImGui::PopTextWrapPos();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(avcodec_license());
    ImGui::PopTextWrapPos();
    ImGui::Separator();
    ImGui::Text("libavformat: %s", LIBAVFORMAT_IDENT);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(avformat_configuration());
    ImGui::PopTextWrapPos();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(avformat_license());
    ImGui::PopTextWrapPos();
    ImGui::Separator();
    ImGui::Text("libavdevice: %s", LIBAVDEVICE_IDENT);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(avdevice_configuration());
    ImGui::PopTextWrapPos();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(avdevice_license());
    ImGui::PopTextWrapPos();
    ImGui::Separator();
    ImGui::Text("ImGui: %s", IMGUI_VERSION);
    ImGui::End();
}

static void draw_help(bool *p_open)
{
    const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration |
                                          ImGuiWindowFlags_AlwaysAutoResize |
                                          ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoNav |
                                          ImGuiWindowFlags_HorizontalScrollbar |
                                          ImGuiWindowFlags_NoFocusOnAppearing |
                                          ImGuiWindowFlags_NoMove;
    const int align = 555;

    ImGui::SetNextWindowPos(ImVec2(display_w/2, display_h/2), 0, ImVec2(0.5, 0.5));
    ImGui::SetNextWindowBgAlpha(help_alpha);
    ImGui::SetNextWindowFocus();

    if (!ImGui::Begin("##Help", p_open, window_flags)) {
        ImGui::End();
        return;
    }

    ImGui::Separator();
    ImGui::Separator();
    ImGui::TextUnformatted("Global Keys:");
    ImGui::Separator();
    ImGui::TextUnformatted("Show Help:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("F1");
    ImGui::Separator();
    ImGui::TextUnformatted("Jump to FilterGraph Editor:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("F2");
    ImGui::Separator();
    ImGui::TextUnformatted("Jump to FilterGraph Commands Window:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("F3");
    ImGui::Separator();
    ImGui::TextUnformatted("Jump to FilterGraph Dump Window:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("F4");
    ImGui::Separator();
    ImGui::TextUnformatted("Jump to FilterGraph Log Window:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("F5");
    ImGui::Separator();
    ImGui::TextUnformatted("Jump to FilterGraph Record Window:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("F6");
    ImGui::Separator();
    ImGui::TextUnformatted("Show Version/Configuration/License:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("F12");
    ImGui::Separator();
    ImGui::TextUnformatted("Toggle Console:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Escape");
    ImGui::Separator();
    ImGui::Separator();
    ImGui::Separator();
    ImGui::TextUnformatted("FilterGraph Editor Keys:");
    ImGui::Separator();
    ImGui::TextUnformatted("Add New Filter:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("A");
    ImGui::Separator();
    ImGui::TextUnformatted("Auto Connect Filter Outputs to Sinks:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Shift + A");
    ImGui::Separator();
    ImGui::TextUnformatted("Remove Selected Filters:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Shift + X");
    ImGui::Separator();
    ImGui::TextUnformatted("Remove Selected Links:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("X");
    ImGui::Separator();
    ImGui::TextUnformatted("Clone Selected Filters:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Shift + C");
    ImGui::Separator();
    ImGui::TextUnformatted("Configure Graph:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Ctrl + Enter");
    ImGui::Separator();
    ImGui::TextUnformatted("Record Configured Graph:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Ctrl + R");
    ImGui::Separator();
    ImGui::Separator();
    ImGui::Separator();
    ImGui::TextUnformatted("Video/Audio FilterGraph Outputs:");
    ImGui::Separator();
    ImGui::TextUnformatted("Pause playback:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Space");
    ImGui::Separator();
    ImGui::TextUnformatted("Toggle fullscreen:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("F");
    ImGui::Separator();
    ImGui::TextUnformatted("Toggle zooming:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Z");
    ImGui::Separator();
    ImGui::TextUnformatted("Framestep forward:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("'.'");
    ImGui::Separator();
    ImGui::TextUnformatted("Toggle OSD:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("O");
    ImGui::Separator();
    ImGui::TextUnformatted("Toggle OSD for all outputs:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Shift + O");
    ImGui::Separator();
    ImGui::TextUnformatted("Jump to next Video output:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Ctrl + Tab");
    ImGui::Separator();
    ImGui::TextUnformatted("Jump to next Audio output:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Alt + Tab");
    ImGui::Separator();
    ImGui::TextUnformatted("Jump to #numbered Video output:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Ctrl + <number>");
    ImGui::Separator();
    ImGui::TextUnformatted("Jump to #numbered Audio output:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Alt + <number>");
    ImGui::Separator();
    ImGui::TextUnformatted("Toggle Audio mute:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("M");
    ImGui::Separator();
    ImGui::TextUnformatted("Toggle Audio mute for all outputs:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Shift + M");
    ImGui::Separator();
    ImGui::TextUnformatted("Show Extended Info:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("I");
    ImGui::Separator();
    ImGui::TextUnformatted("Show Full Extended Info:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Shift + I");
    ImGui::Separator();
    ImGui::TextUnformatted("Exit from output:");
    ImGui::SameLine(align);
    ImGui::TextUnformatted("Shift + Q");
    ImGui::Separator();
    ImGui::End();
}

static void add_filter_node(const AVFilter *filter, ImVec2 pos)
{
    FilterNode node = {0};

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
    node.imported_id = false;
    node.have_exports = false;
    node.show_exports = false;

    filter_nodes.push_back(node);
}

static bool is_source_filter(const AVFilter *filter);
static ImVec2 find_node_spot(ImVec2 start);

static void importfile_filter_graph(const char *file_name)
{
    FILE *file = fopen(file_name, "r");
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

    separators.clear();

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
    label2edge.clear();
    filter2edge.clear();
    filter_links.clear();
    filter_nodes.clear();

    for (unsigned i = 0; i < filters.size(); i++) {
        FilterNode node = {0};
        std::pair <int, int> p = filters[i];
        std::string filter_name;
        std::string instance_name;
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
        if (!node.filter_name) {
            av_log(NULL, AV_LOG_ERROR, "Could not get filter name.\n");
            goto error;
        }
        std::istringstream full_name(node.filter_name);
        std::getline(full_name, filter_name, '@');
        std::getline(full_name, instance_name, '@');
        if (!filter_name.empty())
            node.filter = avfilter_get_by_name(filter_name.c_str());
        if (!node.filter) {
            av_log(NULL, AV_LOG_ERROR, "Could not get filter by name: %s.\n", node.filter_name);
            goto error;
        }
        node.imported_id = instance_name.length() > 0;
        node.filter_label = node.imported_id && (!instance_name.empty()) ? av_asprintf("%s@%s", node.filter->name, instance_name.c_str()) : av_asprintf("%s@%d", node.filter->name, node.id);
        node.filter_options = opts;
        node.ctx_options = NULL;
        node.probe = avfilter_graph_alloc_filter(probe_graph, node.filter, "probe");
        node.ctx = NULL;
        node.pos = find_node_spot(ImVec2(300, 300));
        node.colapsed = false;
        node.have_exports = false;
        node.show_exports = false;
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

struct {
    bool operator()(int a, int b) const
    {
        const std::pair<int, int> pa = filter_links[a];
        const std::pair<int, int> pb = filter_links[b];
        const int paa = pa.first;
        const int pab = pa.second;
        const int pba = pb.first;
        const int pbb = pb.second;
        int pia, pib;

        if (!edge2pad[paa].is_output)
            pia = edge2pad[paa].pad_index;
        if (!edge2pad[pab].is_output)
            pia = edge2pad[pab].pad_index;

        if (!edge2pad[pba].is_output)
            pib = edge2pad[pba].pad_index;
        if (!edge2pad[pbb].is_output)
            pib = edge2pad[pbb].pad_index;

        return pia > pib;
    }
} inputPads;

struct {
    bool operator()(int a, int b) const
    {
        const std::pair<int, int> pa = filter_links[a];
        const std::pair<int, int> pb = filter_links[b];
        const int paa = pa.first;
        const int pab = pa.second;
        const int pba = pb.first;
        const int pbb = pb.second;
        int pia, pib;

        if (edge2pad[paa].is_output)
            pia = edge2pad[paa].pad_index;
        if (edge2pad[pab].is_output)
            pia = edge2pad[pab].pad_index;

        if (edge2pad[pba].is_output)
            pib = edge2pad[pba].pad_index;
        if (edge2pad[pbb].is_output)
            pib = edge2pad[pbb].pad_index;

        return pia > pib;
    }
} outputPads;

static void export_filter_graph(char **out, size_t *out_size)
{
    std::vector<bool> visited;
    std::vector<unsigned> to_visit;
    std::vector<unsigned> pads;
    AVBPrint buf;
    bool first = true;

    av_bprint_init(&buf, 512, AV_BPRINT_SIZE_UNLIMITED);

    visited.resize(filter_nodes.size());

    for (size_t i = 0; i < filter_nodes.size(); i++) {
        if (is_source_filter(filter_nodes[i].filter))
            to_visit.push_back(i);
    }

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
                pads.push_back(i);
            }

            std::sort(pads.begin(), pads.end(), inputPads);
            while (pads.size() > 0) {
                unsigned pad = pads.back();

                pads.pop_back();
                av_bprintf(&buf, "[e%d]", pad);
            }

            if (filter_nodes[node].imported_id)
                av_bprintf(&buf, "%s", filter_nodes[node].filter_label);
            else
                av_bprintf(&buf, "%s@%d", filter_nodes[node].filter_name, filter_nodes[node].id);
            av_freep(&filter_nodes[node].filter_options);
            av_opt_serialize(filter_nodes[node].ctx->priv, AV_OPT_FLAG_FILTERING_PARAM, AV_OPT_SERIALIZE_SKIP_DEFAULTS,
                             &filter_nodes[node].filter_options, '=', ':');
            if (filter_nodes[node].filter_options && (strlen(filter_nodes[node].filter_options) > 0))
                av_bprintf(&buf, "=%s", filter_nodes[node].filter_options);
            av_freep(&filter_nodes[node].filter_options);

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
                pads.push_back(i);
            }

            std::sort(pads.begin(), pads.end(), outputPads);
            while (pads.size() > 0) {
                unsigned pad = pads.back();

                pads.pop_back();
                av_bprintf(&buf, "[e%d]", pad);
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

static void exportfile_filter_graph(const char *file_name)
{
    size_t out_size = 0;
    char *out = NULL;

    export_filter_graph(&out, &out_size);

    if (out && out_size > 0) {
        FILE *script_file = fopen(file_name, "w");

        if (script_file) {
            fwrite(out, 1, out_size, script_file);
            fclose(script_file);
        } else {
            av_log(NULL, AV_LOG_ERROR, "Cannot open '%s' script.\n", file_name);
        }
        av_freep(&out);
        out_size = 0;
    }
}

typedef struct ConsoleData {
    void *opaque;
    int pos;
} ConsoleData;

static int console_callback(ImGuiInputTextCallbackData *data)
{
    if (data->EventFlag == ImGuiInputTextFlags_CallbackCompletion) {
        if (!strncmp(data->Buf, "a ", 2)) {
            ConsoleData *console_data = (ConsoleData *)data->UserData;

            if (data->CursorPos > 3) {
                const AVFilter *filter;

                do {
                    filter = av_filter_iterate(&console_data->opaque);
                    if (filter == NULL)
                        break;

                    const int name_len = strlen(filter->name);
                    if (name_len <= console_data->pos-2)
                        continue;

                    if (!strncmp(data->Buf+2, filter->name, console_data->pos-2)) {
                        int count = std::max(data->BufTextLen - console_data->pos, 0);

                        data->DeleteChars(console_data->pos, count);
                        data->InsertChars(console_data->pos, filter->name+(console_data->pos-2));
                        break;
                    }
                } while (filter);

                if (filter == NULL) {
                    int count = std::max(data->BufTextLen - console_data->pos, 0);

                    data->DeleteChars(console_data->pos, count);
                    console_data->opaque = NULL;
                }
            }
        }
    } else if (data->EventFlag == ImGuiInputTextFlags_CallbackHistory) {
        if (data->EventKey == ImGuiKey_UpArrow) {
        } else if (data->EventKey == ImGuiKey_DownArrow) {
        }
    } else if (data->EventFlag == ImGuiInputTextFlags_CallbackEdit) {
        ConsoleData *console_data = (ConsoleData *)data->UserData;

        console_data->pos = data->CursorPos;
    }

    return 0;
}

ConsoleData console_data = { 0 };

static void draw_console(bool *p_open)
{
    char input_line[4096] = { 0 };
    const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration |
                                          ImGuiWindowFlags_NoSavedSettings |
                                          ImGuiWindowFlags_NoNav |
                                          ImGuiWindowFlags_NoMove;

    ImGui::SetNextWindowPos(ImVec2(0, display_h - 35));
    ImGui::SetNextWindowSize(ImVec2(display_w, 30));
    ImGui::SetNextWindowBgAlpha(console_alpha);
    ImGui::SetNextWindowFocus();

    if (!ImGui::Begin("##Console", p_open, window_flags)) {
        ImGui::End();
        return;
    }

    ImGuiInputTextFlags input_text_flags = ImGuiInputTextFlags_EnterReturnsTrue |
                                           ImGuiInputTextFlags_CallbackCompletion |
                                           ImGuiInputTextFlags_CallbackEdit;

    ImGui::SetKeyboardFocusHere();
    ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0,   0, 0, 200));
    ImGui::PushStyleColor(ImGuiCol_Text,    IM_COL32(0, 255, 0, 200));
    if (ImGui::InputText("##>", input_line, IM_ARRAYSIZE(input_line), input_text_flags, console_callback, &console_data)) {
        if (!strncmp(input_line, "a ", 2) && filter_graph_is_valid == false) {
            const AVFilter *filter = avfilter_get_by_name(input_line + 2);

            if (filter)
                add_filter_node(filter, ImVec2(0, 0));
        }

        if (!strncmp(input_line, "i ", 2) && filter_graph_is_valid == false) {
            const char *file_name = input_line + 2;

            if (file_name)
                importfile_filter_graph(file_name);
        }

        if (!strncmp(input_line, "e ", 2) && filter_graph_is_valid == true) {
            const char *file_name = input_line + 2;

            if (file_name)
                exportfile_filter_graph(file_name);
        }
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();

    ImGui::End();
}

static void draw_osd(BufferSink *sink, int width, int height)
{
    char osd_text[1024];

    snprintf(osd_text, sizeof(osd_text), "FRAME: %ld | SIZE: %dx%d | TIME: %.5f | SPEED: %011.5f | FPS: %d/%d (%.5f)",
             sink->frame_number - 1,
             width, height,
             av_q2d(sink->time_base) * sink->pts,
             sink->speed,
             sink->frame_rate.num, sink->frame_rate.den, av_q2d(sink->frame_rate));

    if (sink->fullscreen) {
        ImVec2 max_size = ImGui::GetIO().DisplaySize;
        ImVec2 tsize = ImGui::CalcTextSize(osd_text);
        ImVec2 start_pos = ImVec2(std::min(max_size.x * osd_fullscreen_pos[0], max_size.x - tsize.x - 25), std::min(max_size.y * osd_fullscreen_pos[1], max_size.y - tsize.y - 25));
        ImVec2 stop_pos = ImVec2(std::min(start_pos.x + tsize.x + 25, max_size.x), std::min(start_pos.y + tsize.y + 25, max_size.y));
        ImGui::GetWindowDrawList()->AddRectFilled(start_pos, stop_pos,
                                                  ImGui::GetColorU32(ImGuiCol_WindowBg, osd_alpha));
        ImGui::SetCursorPos(ImVec2(std::min(start_pos.x + 12, max_size.x - tsize.x - 12), std::min(start_pos.y + 12, max_size.y - tsize.y - 12)));
        ImGui::TextUnformatted(osd_text);
    } else {
        ImGui::TextWrapped("%s", osd_text);
    }
}

static void update_frame_info(FrameInfo *frame_info, const AVFrame *frame)
{
    if (!ImGui::IsKeyDown(ImGuiKey_I))
        return;

    frame_info->width = frame->width;
    frame_info->height = frame->height;
    frame_info->nb_samples = frame->nb_samples;
    frame_info->format = frame->format;
    frame_info->key_frame = !!(frame->flags & AV_FRAME_FLAG_KEY);
    frame_info->pict_type = frame->pict_type;
    frame_info->sample_aspect_ratio = frame->sample_aspect_ratio;
    frame_info->pts = frame->pts;
    frame_info->time_base = frame->time_base;
    frame_info->interlaced_frame = !!(frame->flags & AV_FRAME_FLAG_INTERLACED);
    frame_info->top_field_first = !!(frame->flags & AV_FRAME_FLAG_TOP_FIELD_FIRST);
    frame_info->sample_rate = frame->sample_rate;
    av_channel_layout_copy(&frame_info->ch_layout, &frame->ch_layout);
    frame_info->color_range = frame->color_range;
    frame_info->color_primaries = frame->color_primaries;
    frame_info->color_trc = frame->color_trc;
    frame_info->colorspace = frame->colorspace;
    frame_info->chroma_location = frame->chroma_location;
    frame_info->duration = frame->duration;
    frame_info->crop_top = frame->crop_top;
    frame_info->crop_bottom = frame->crop_bottom;
    frame_info->crop_left = frame->crop_left;
    frame_info->crop_right = frame->crop_right;
}

static void draw_frame(bool *p_open, ring_item_t item, BufferSink *sink)
{
    ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
    GLuint *texture = NULL;
    int width, height;
    bool style = false;

    if (item.frame)
        update_frame_info(&sink->frame_info, item.frame);

    if (!*p_open)
        return;

    if (item.frame) {
        texture = &item.id.u.v;
        load_frame(texture, &width, &height, item.frame, sink);
    } else if (sink->texture &&
               sink->width > 0 &&
               sink->height > 0) {
        texture = &sink->texture;
        width = sink->width;
        height = sink->height;
    } else {
        return;
    }

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

    if (ImGui::IsKeyDown((ImGuiKey)(ImGuiKey_0 + av_clip(sink->id,0,9))) && ImGui::GetIO().KeyCtrl)
        focus_buffersink_window = av_clip(sink->id,0,9);

    if (focus_buffersink_window == sink->id)
        ImGui::SetNextWindowFocus();

    ImGui::SetNextWindowBgAlpha(sink_alpha);
    if (!ImGui::Begin(sink->label, p_open, flags)) {
        ImGui::End();
        return;
    }

    if (sink->fullscreen == false)
        sink->window_pos = ImGui::GetWindowPos();

    if (ImGui::IsWindowFocused()) {
        if (focus_buffersink_window == sink->id) {
            last_buffersink_window = focus_buffersink_window;
            focus_buffersink_window = UINT_MAX;
        }
        if (ImGui::IsKeyReleased(ImGuiKey_F))
            sink->fullscreen = !sink->fullscreen;
        if (ImGui::IsKeyReleased(ImGuiKey_Space))
            paused = !paused;
        framestep = ImGui::IsKeyPressed(ImGuiKey_Period);
        if (framestep)
            paused = true;
        if (ImGui::IsKeyReleased(ImGuiKey_M) && (ImGui::GetIO().KeyShift))
            muted_all = !muted_all;
        if (ImGui::IsKeyDown(ImGuiKey_Q) && ImGui::GetIO().KeyShift) {
            last_abuffersink_window = 0;
            last_buffersink_window = 0;
            show_abuffersink_window = false;
            show_buffersink_window = false;
            filter_graph_is_valid = false;
        }
        if (ImGui::IsKeyReleased(ImGuiKey_O)) {
            if (ImGui::GetIO().KeyShift)
                show_osd_all = !show_osd_all;
            else
                sink->show_osd = !sink->show_osd;
        }
    }

    if (sink->fullscreen && texture) {
        ImGui::GetWindowDrawList()->AddImage(*texture, ImVec2(0.f, 0.f),
                                             ImGui::GetWindowSize(),
                                             ImVec2(0.f, 0.f), ImVec2(1.f, 1.f), IM_COL32_WHITE);
    } else if (texture) {
        ImGui::Image(*texture, ImVec2(width, height));
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
        ImGui::Image(*texture, ImVec2(region_sz * zoom, region_sz * zoom), uv0, uv1, tint_col, border_col);
        ImGui::EndTooltip();
    }

    if (sink->show_osd ^ show_osd_all)
        draw_osd(sink, width, height);

    if (style) {
        ImGui::PopStyleVar();
        ImGui::PopStyleVar();
    }

    ImGui::End();
}

static void draw_aosd(BufferSink *sink)
{
    ALfloat sec_offset;
    ALint queued;

    alGetSourcei(sink->source, AL_BUFFERS_QUEUED, &queued);
    alGetSourcef(sink->source, AL_SEC_OFFSET, &sec_offset);

    if (sink->fullscreen) {
        char osd_text[1024];

        snprintf(osd_text, sizeof(osd_text), "FRAME: %ld | SIZE: %d | TIME: %.5f | SPEED: %011.5f | RATE: %d | QUEUE: %d",
                 sink->frame_number - 1, sink->frame_nb_samples,
                 (sink->pts != AV_NOPTS_VALUE) ? (av_q2d(sink->time_base) * sink->pts) + sec_offset : NAN,
                 sink->speed,
                 sink->sample_rate,
                 queued);

        ImVec2 max_size = ImGui::GetIO().DisplaySize;
        ImVec2 tsize = ImGui::CalcTextSize(osd_text);
        ImVec2 start_pos = ImVec2(std::min(max_size.x * osd_fullscreen_pos[0], max_size.x - tsize.x - 25), std::min(max_size.y * osd_fullscreen_pos[1], max_size.y - tsize.y - 25));
        ImVec2 stop_pos = ImVec2(std::min(start_pos.x + tsize.x + 25, max_size.x), std::min(start_pos.y + tsize.y + 25, max_size.y));
        ImGui::GetWindowDrawList()->AddRectFilled(start_pos, stop_pos,
                                                  ImGui::GetColorU32(ImGuiCol_WindowBg, osd_alpha));
        ImGui::SetCursorPos(ImVec2(std::min(start_pos.x + 12, max_size.x - tsize.x - 12), std::min(start_pos.y + 12, max_size.y - tsize.y - 12)));
        ImGui::TextUnformatted(osd_text);
    } else {
        ImGui::Text("FRAME: %ld", sink->frame_number - 1);
        ImGui::Text("SIZE:  %d", sink->frame_nb_samples);
        ImGui::Text("TIME:  %.5f", sink->pts != AV_NOPTS_VALUE ? av_q2d(sink->time_base) * sink->pts + sec_offset : NAN);
        ImGui::Text("SPEED: %011.5f", sink->speed);
        ImGui::Text("RATE:  %d", sink->sample_rate);
        ImGui::Text("QUEUE: %d", queued);
    }
}

static void load_aframe(BufferSink *sink, AVFrame *frame)
{
    if (frame->format == AV_SAMPLE_FMT_FLTP) {
        const float *src = (const float *)frame->extended_data[0];

        if (src && frame->nb_samples > 0) {
            float min = FLT_MAX, max = -FLT_MAX;

            for (int n = 0; n < frame->nb_samples; n++) {
                max = std::max(max, src[n]);
                min = std::min(min, src[n]);
            }

            sink->frame_number++;
            sink->frame_nb_samples = frame->nb_samples;
            sink->samples[sink->sample_index++] = max;
            sink->samples[sink->sample_index++] = min;
            if (sink->sample_index >= sink->nb_samples)
                sink->sample_index = 0;
        }
    } else {
        const double *src = (const double *)frame->extended_data[0];

        if (src && frame->nb_samples > 0) {
            double min = DBL_MAX, max = -DBL_MAX;

            for (int n = 0; n < frame->nb_samples; n++) {
                max = std::max(max, src[n]);
                min = std::min(min, src[n]);
            }

            sink->frame_number++;
            sink->frame_nb_samples = frame->nb_samples;
            sink->samples[sink->sample_index++] = max;
            sink->samples[sink->sample_index++] = min;
            if (sink->sample_index >= sink->nb_samples)
                sink->sample_index = 0;
        }
    }
}

static void draw_aframe(bool *p_open, ring_item_t item, BufferSink *sink)
{
    ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
    bool style = false;

    if (item.frame)
        update_frame_info(&sink->frame_info, item.frame);

    if (!*p_open)
        return;

    if (item.frame && sink->pts != item.frame->pts) {
        sink->pts = item.frame->pts;
        load_aframe(sink, item.frame);
    }

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
        if (sink->have_window_pos == true) {
            ImGui::SetNextWindowPos(sink->window_pos);
            sink->have_window_pos = false;
        }
    }

    if (ImGui::IsKeyDown((ImGuiKey)(ImGuiKey_0 + av_clip(sink->id,0,9))) && ImGui::GetIO().KeyAlt)
        focus_abuffersink_window = av_clip(sink->id,0,9);

    if (focus_abuffersink_window == sink->id)
        ImGui::SetNextWindowFocus();

    ImGui::SetNextWindowBgAlpha(sink_alpha);
    if (!ImGui::Begin(sink->label, p_open, flags)) {
        ImGui::End();
        return;
    }

    if (sink->fullscreen == false)
        sink->window_pos = ImGui::GetWindowPos();

    if (ImGui::IsWindowFocused()) {
        if (focus_abuffersink_window == sink->id) {
            last_abuffersink_window = focus_abuffersink_window;
            focus_abuffersink_window = UINT_MAX;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", sink->description);
        if (ImGui::IsKeyReleased(ImGuiKey_F))
            sink->fullscreen = !sink->fullscreen;
        if (ImGui::IsKeyReleased(ImGuiKey_Space))
            paused = !paused;
        framestep = ImGui::IsKeyPressed(ImGuiKey_Period);
        if (framestep)
            paused = true;
        if (ImGui::IsKeyReleased(ImGuiKey_M)) {
            if (ImGui::GetIO().KeyShift)
                muted_all = !muted_all;
            else
                sink->muted = !sink->muted;
        }
        if (ImGui::IsKeyDown(ImGuiKey_Q) && ImGui::GetIO().KeyShift) {
            last_abuffersink_window = 0;
            last_buffersink_window = 0;
            show_abuffersink_window = false;
            show_buffersink_window = false;
            filter_graph_is_valid = false;
        }
        if (ImGui::IsKeyReleased(ImGuiKey_O)) {
            if (ImGui::GetIO().KeyShift)
                show_osd_all = !show_osd_all;
            else
                sink->show_osd = !sink->show_osd;
        }
    }

    if (sink->fullscreen) {
        const char *label = (sink->muted ^ muted_all) ? "MUTED" : NULL;
        ImVec2 window_size = { -1, -1 };

        ImGui::PlotLines("##Audio Samples", sink->samples, sink->nb_samples, 0, label, -audio_sample_range[0], audio_sample_range[1], window_size);
    } else {
        ImVec2 window_size = { audio_window_size[0], audio_window_size[1] };
        const char *label = (sink->muted ^ muted_all) ? "MUTED" : NULL;

        ImGui::PlotLines("##Audio Samples", sink->samples, sink->nb_samples, 0, label, -audio_sample_range[0], audio_sample_range[1], window_size);
    }

    if (sink->show_osd ^ show_osd_all)
        draw_aosd(sink);

    if (sink->fullscreen == false) {
        if (ImGui::DragFloat("Gain", &sink->gain, 0.01f, 0.f, 2.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput))
            alSourcef(sink->source, AL_GAIN, sink->gain);
        if (ImGui::DragFloat3("Position", sink->position, 0.01f, -1.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput))
            alSource3f(sink->source, AL_POSITION, sink->position[0], sink->position[1], sink->position[2]);
    }

    if (style) {
        ImGui::PopStyleVar();
        ImGui::PopStyleVar();
    }

    ImGui::End();
}

static void queue_sound(BufferSink *sink, ring_item_t item)
{
    AVFrame *frame = item.frame;
    ALuint aid = item.id.u.a;
    const size_t sample_size = (frame->format == AV_SAMPLE_FMT_FLTP) ? sizeof(float) : sizeof(double);

    alSourcef(sink->source, AL_GAIN, sink->gain * !(sink->muted ^ muted_all));

    alBufferData(aid, sink->format, frame->extended_data[0],
                 (ALsizei)frame->nb_samples * sample_size, frame->sample_rate);
    alSourceQueueBuffers(sink->source, 1, &aid);
}

static bool query_ranges(void *obj, const AVOption *opt,
                         double *min, double *max)
{
    AVOptionType type = (AVOptionType)(opt->type & (~AV_OPT_TYPE_FLAG_ARRAY));

    switch (type) {
        case AV_OPT_TYPE_INT:
        case AV_OPT_TYPE_UINT:
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
        avfilter_filter_pad_count(filter, 1) == 1 &&
        !(filter->flags & AVFILTER_FLAG_DYNAMIC_INPUTS) &&
        !(filter->flags & AVFILTER_FLAG_DYNAMIC_OUTPUTS))
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

static bool is_complex_audio_filter(const AVFilter *filter)
{
    if (is_complex_filter(filter)) {
        for (unsigned i = 0; i < avfilter_filter_pad_count(filter, 0); i++) {
            if (avfilter_pad_get_type(filter->inputs, i) != AVMEDIA_TYPE_AUDIO)
                return false;
        }

        for (unsigned i = 0; i < avfilter_filter_pad_count(filter, 1); i++) {
            if (avfilter_pad_get_type(filter->outputs, i) != AVMEDIA_TYPE_AUDIO)
                return false;
        }

        return true;
    }

    return false;
}

static bool is_complex_video_filter(const AVFilter *filter)
{
    if (is_complex_filter(filter)) {
        for (unsigned i = 0; i < avfilter_filter_pad_count(filter, 0); i++) {
            if (avfilter_pad_get_type(filter->inputs, i) != AVMEDIA_TYPE_VIDEO)
                return false;
        }

        for (unsigned i = 0; i < avfilter_filter_pad_count(filter, 1); i++) {
            if (avfilter_pad_get_type(filter->outputs, i) != AVMEDIA_TYPE_VIDEO)
                return false;
        }

        return true;
    }

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

        node->have_exports |= !!(opt->flags & AV_OPT_FLAG_EXPORT);

        if (opt->flags & AV_OPT_FLAG_READONLY)
            continue;

        if (!query_ranges((void *)obj, opt, &min, &max))
            continue;

        if ((AVOptionType)(opt->type & (~AV_OPT_TYPE_FLAG_ARRAY)) == AV_OPT_TYPE_CONST)
            continue;

        if (opt->type & AV_OPT_TYPE_FLAG_ARRAY) {
            AVOptionType type = (AVOptionType)(opt->type & (~AV_OPT_TYPE_FLAG_ARRAY));
            unsigned int nb_elems = 0;

            av_opt_get_array_size(av_class, opt->name, 0, &nb_elems);
            if (nb_elems > 0) {
                switch (type) {
                    case AV_OPT_TYPE_DOUBLE:
                        {
                            double *value = (double *)av_calloc(nb_elems, sizeof(*value));

                            if (!value)
                                break;

                            if (av_opt_get_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value) < 0) {
                                av_freep(&value);
                                break;
                            }

                            ImGui::SetNextItemWidth(200.f);
                            if (ImGui::DragScalarN(opt->name, ImGuiDataType_Double, value, nb_elems,
                                                   (max-min)/200.0, &min, &max, "%f", ImGuiSliderFlags_AlwaysClamp)) {
                                av_opt_set_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value);
                            }
                            av_freep(&value);
                        }
                        break;
                    case AV_OPT_TYPE_FLOAT:
                        {
                            float *value = (float *)av_calloc(nb_elems, sizeof(*value));
                            float fmin = min;
                            float fmax = max;

                            if (!value)
                                break;

                            if (av_opt_get_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value) < 0) {
                                av_freep(&value);
                                break;
                            }

                            ImGui::SetNextItemWidth(200.f);
                            if (ImGui::DragScalarN(opt->name, ImGuiDataType_Float, value, nb_elems,
                                                   (fmax-fmin)/200.f, &fmin, &fmax, "%f", ImGuiSliderFlags_AlwaysClamp)) {
                                av_opt_set_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value);
                            }
                            av_freep(&value);
                        }
                        break;
                    case AV_OPT_TYPE_BOOL:
                    case AV_OPT_TYPE_INT:
                        {
                            int32_t *value = (int32_t *)av_calloc(nb_elems, sizeof(*value));
                            int32_t imin = min;
                            int32_t imax = max;

                            if (!value)
                                break;

                            if (av_opt_get_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value) < 0) {
                                av_freep(&value);
                                break;
                            }

                            ImGui::SetNextItemWidth(200.f);
                            if (ImGui::DragScalarN(opt->name, ImGuiDataType_S32, value, nb_elems,
                                                   1, &imin, &imax, "%d", ImGuiSliderFlags_AlwaysClamp)) {
                                av_opt_set_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value);
                            }
                            av_freep(&value);
                        }
                        break;
                    case AV_OPT_TYPE_FLAGS:
                    case AV_OPT_TYPE_UINT:
                        {
                            uint32_t *value = (uint32_t *)av_calloc(nb_elems, sizeof(*value));
                            uint32_t umin = min;
                            uint32_t umax = max;

                            if (!value)
                                break;

                            if (av_opt_get_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value) < 0) {
                                av_freep(&value);
                                break;
                            }

                            ImGui::SetNextItemWidth(200.f);
                            if (ImGui::DragScalarN(opt->name, ImGuiDataType_U32, value, nb_elems,
                                                   1, &umin, &umax, "%u", ImGuiSliderFlags_AlwaysClamp)) {
                                av_opt_set_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value);
                            }
                            av_freep(&value);
                        }
                        break;
                    case AV_OPT_TYPE_COLOR:
                        {
                            uint32_t *value = (uint32_t *)av_calloc(nb_elems, sizeof(*value));

                            if (!value)
                                break;

                            if (av_opt_get_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value) < 0) {
                                av_freep(&value);
                                break;
                            }

                            for (unsigned int i = 0; i < nb_elems; i++) {
                                char label[1024] = {0};
                                uint8_t icol[4];
                                float col[4];

                                snprintf(label, sizeof(label), "%s.%u", opt->name, i);
                                AV_WN32(icol, value[i]);

                                col[0] = icol[0] / 255.f;
                                col[1] = icol[1] / 255.f;
                                col[2] = icol[2] / 255.f;
                                col[3] = icol[3] / 255.f;

                                ImGui::SetNextItemWidth(200.f);
                                ImGui::ColorEdit4(label, col, ImGuiColorEditFlags_NoDragDrop);

                                icol[0] = col[0] * 255.f;
                                icol[1] = col[1] * 255.f;
                                icol[2] = col[2] * 255.f;
                                icol[3] = col[3] * 255.f;

                                value[i] = AV_RN32(icol);
                            }
                            av_opt_set_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value);
                            av_freep(&value);
                        }
                        break;
                    case AV_OPT_TYPE_STRING:
                        {
                            char **value = (char **)av_calloc(nb_elems, sizeof(*value));
                            bool new_value = false;

                            if (!value)
                                break;

                            if (av_opt_get_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value) < 0) {
                                av_freep(&value);
                                break;
                            }

                            for (unsigned int i = 0; i < nb_elems; i++) {
                                char label[1024] = {0};
                                char string[1024] = {0};

                                snprintf(label, sizeof(label), "%s.%u", opt->name, i);

                                ImGui::SetNextItemWidth(200.f);
                                if (value[i])
                                    memcpy(string, value[i], std::min(sizeof(string)-1, strlen(value[i])));
                                if (ImGui::InputText(label, string, IM_ARRAYSIZE(string))) {
                                    av_freep(&value[i]);
                                    value[i] = av_strdup(string);
                                    new_value = true;
                                }
                            }

                            if (new_value == true)
                                av_opt_set_array(av_class, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, value);
                            for (unsigned int i = 0; i < nb_elems; i++) {
                                av_freep(&value[i]);
                            }
                            av_freep(&value);
                        }
                        break;
                    default:
                        break;
                }
            }

            if (ImNodes::IsNodeSelected(node->edge) && ImGui::IsItemHovered() && opt->type != AV_OPT_TYPE_CONST)
                ImGui::SetTooltip("%s", opt->help);

            if (opt->default_val.arr) {
                ImGui::SameLine();
                ImGui::PushID(opt->name);
                if ((nb_elems >= opt->default_val.arr->size_max) &&
                    opt->default_val.arr->size_max > 0)
                    ImGui::BeginDisabled();
                if (ImGui::SmallButton("+")) {
                    OptStorage new_element = { 0 };

                    switch (type) {
                    case AV_OPT_TYPE_FLOAT:
                        new_element.u.flt = (max-min)/2;
                        break;
                    case AV_OPT_TYPE_DOUBLE:
                        new_element.u.dbl = (max-min)/2;
                        break;
                    case AV_OPT_TYPE_UINT:
                        new_element.u.u32 = (max-min)/2;
                        break;
                    case AV_OPT_TYPE_INT:
                        new_element.u.i32 = (max-min)/2;
                        break;
                    case AV_OPT_TYPE_INT64:
                        new_element.u.i64 = (max-min)/2;
                        break;
                    case AV_OPT_TYPE_UINT64:
                        new_element.u.u64 = (max-min)/2;
                        break;
                    case AV_OPT_TYPE_BOOL:
                        new_element.u.i32 = (max-min)/2;
                        break;
                    case AV_OPT_TYPE_FLAGS:
                        new_element.u.u32 = (max-min)/2;
                        break;
                    case AV_OPT_TYPE_STRING:
                        new_element.u.str = av_strdup("<empty>");
                        break;
                    default:
                        break;
                    }

                    av_opt_set_array(av_class, opt->name, 0, nb_elems, 1, type, &new_element);
                }
                if ((nb_elems >= opt->default_val.arr->size_max) &&
                    opt->default_val.arr->size_max > 0)
                    ImGui::EndDisabled();
                ImGui::PopID();
            }

            if (opt->default_val.arr) {
                ImGui::SameLine();
                ImGui::PushID(opt->name);
                if (nb_elems <= opt->default_val.arr->size_min)
                    ImGui::BeginDisabled();
                if (ImGui::SmallButton("-")) {
                    av_opt_set_array(av_class, opt->name, 0, nb_elems-1, 1, type, NULL);
                }
                if (nb_elems <= opt->default_val.arr->size_min)
                    ImGui::EndDisabled();
                ImGui::PopID();
            }
        } else {
            switch (opt->type) {
                case AV_OPT_TYPE_INT64:
                    {
                        int64_t value;
                        int64_t smin = min;
                        int64_t smax = max;
                        if (av_opt_get_int(av_class, opt->name, 0, &value))
                            break;
                        ImGui::SetNextItemWidth(200.f);
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
                        ImGui::SetNextItemWidth(200.f);
                        if (ImGui::DragScalar(opt->name, ImGuiDataType_U64, &value, (umax-umin)/200.f, &umin, &umax, "%lu", ImGuiSliderFlags_AlwaysClamp)) {
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
                        ImGui::SetNextItemWidth(200.f);
                        if (ImGui::DragScalar(opt->name, ImGuiDataType_Double, &dvalue, 0.1f, &min, &max, "%f", ImGuiSliderFlags_AlwaysClamp)) {
                            value = dvalue * 1000000.0;
                            av_opt_set_int(av_class, opt->name, value, 0);
                        }
                    }
                    break;
                case AV_OPT_TYPE_FLAGS:
                    {
                        const AVOption *copt = NULL;
                        ImU64 imin = min;
                        ImU64 imax = max;
                        int64_t value;
                        ImU64 uvalue;

                        if (av_opt_get_int(av_class, opt->name, 0, &value))
                            break;

                        uvalue = value;
                        ImGui::SetNextItemWidth(200.f);
                        if (ImGui::DragScalar(opt->name, ImGuiDataType_U64, &uvalue, 1, &imin, &imax, "%d", ImGuiSliderFlags_AlwaysClamp)) {
                            value = uvalue;
                            av_opt_set_int(av_class, opt->name, value, 0);
                        }
                        if (opt->unit) {
                            while ((copt = av_opt_next(obj, copt))) {
                                if (!copt->unit)
                                    continue;
                                if (strcmp(copt->unit, opt->unit) || copt->type != AV_OPT_TYPE_CONST)
                                    continue;

                                ImGui::SetNextItemWidth(200.f);
                                ImGui::CheckboxFlags(copt->name, &uvalue, copt->default_val.i64);
                                if (copt->help) {
                                    ImGui::SameLine();
                                    ImGui::Text("\t\t%s", copt->help);
                                }
                            }
                        }

                        if (value != (int64_t)uvalue) {
                            value = uvalue;
                            av_opt_set_int(av_class, opt->name, value, 0);
                        }
                    }
                    break;
                case AV_OPT_TYPE_BOOL:
                    {
                        int64_t value;
                        int ivalue;
                        int imin = min;
                        int imax = max;
                        if (av_opt_get_int(av_class, opt->name, 0, &value))
                            break;
                        ivalue = value;
                        ImGui::SetNextItemWidth(200.f);
                        if (imax < INT_MAX/2 && imin > INT_MIN/2) {
                            if (ImGui::SliderInt(opt->name, &ivalue, imin, imax, "%d", ImGuiSliderFlags_AlwaysClamp)) {
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
                            char preview_value[20];
                            char combo_name[20];

                            snprintf(combo_name, sizeof(combo_name), "##%s", opt->unit);
                            snprintf(preview_value, sizeof(preview_value), "%ld", value);
                            ImGui::SetNextItemWidth(200.f);
                            if (ImGui::BeginCombo(combo_name, preview_value, 0)) {
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
                        ImGui::SetNextItemWidth(200.f);
                        if (imax < INT_MAX/2 && imin > INT_MIN/2) {
                            if (ImGui::SliderInt(opt->name, &ivalue, imin, imax, "%d", ImGuiSliderFlags_AlwaysClamp)) {
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
                            char preview_value[20];
                            char combo_name[20];

                            snprintf(combo_name, sizeof(combo_name), "##%s", opt->unit);
                            snprintf(preview_value, sizeof(preview_value), "%ld", value);
                            ImGui::SetNextItemWidth(200.f);
                            if (ImGui::BeginCombo(combo_name, preview_value, 0)) {
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
                        ImGui::SetNextItemWidth(200.f);
                        if (ImGui::DragScalar(opt->name, ImGuiDataType_Double, &value, (max-min)/200.f, &min, &max, "%f", ImGuiSliderFlags_AlwaysClamp)) {
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
                        ImGui::SetNextItemWidth(200.f);
                        if (ImGui::DragFloat(opt->name, &fvalue, (fmax-fmin)/200.f, fmin, fmax, "%f", ImGuiSliderFlags_AlwaysClamp)) {
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
                        ImGui::SetNextItemWidth(200.f);
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
                        ImGui::SetNextItemWidth(200.f);
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
                        ImGui::SetNextItemWidth(200.f);
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
                        ImGui::SetNextItemWidth(200.f);
                        if (ImGui::DragInt2(opt->name, irate, 1, -8192, 8192)) {
                            snprintf(rate_str, sizeof(rate_str), "%d/%d", irate[0], irate[1]);
                            av_opt_set(av_class, opt->name, rate_str, 0);
                        }
                    }
                    break;
                case AV_OPT_TYPE_PIXEL_FMT:
                    {
                        AVPixelFormat fmt;
                        const char *preview_name;

                        av_opt_get_pixel_fmt(av_class, opt->name, 0, &fmt);
                        ImGui::SetNextItemWidth(200.f);
                        preview_name = av_get_pix_fmt_name(fmt);
                        if (!preview_name)
                            preview_name = "none";

                        if (ImGui::BeginCombo(opt->name, preview_name, 0)) {
                            const AVPixFmtDescriptor *pix_desc = NULL;

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
                    }
                    break;
                case AV_OPT_TYPE_SAMPLE_FMT:
                    {
                        AVSampleFormat fmt;
                        ImGui::SetNextItemWidth(200.f);
                        const char *preview_name;

                        av_opt_get_sample_fmt(av_class, opt->name, 0, &fmt);
                        preview_name = av_get_sample_fmt_name(fmt);
                        if (!preview_name)
                            preview_name = "none";

                        if (ImGui::BeginCombo(opt->name, preview_name, 0)) {
                            const unsigned nb_sample_fmts = sizeof(all_sample_fmts)/sizeof(all_sample_fmts[0]);

                            for (unsigned i = 0; i < nb_sample_fmts; i++) {
                                const bool is_selected = all_sample_fmts[i] == fmt;
                                const char *name = av_get_sample_fmt_name(all_sample_fmts[i]);

                                if (name == NULL)
                                    name = "none";
                                if (ImGui::Selectable(name, is_selected))
                                    av_opt_set_sample_fmt(av_class, opt->name, all_sample_fmts[i], 0);

                                if (is_selected)
                                    ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }
                    }
                    break;
                case AV_OPT_TYPE_COLOR:
                    {
                        float col[4] = { 0.4f, 0.7f, 0.0f, 0.5f };
                        uint8_t icol[4] = { 0 };
                        char new_str[16] = { 0 };
                        uint8_t *old_str = NULL;

                        if (av_opt_get(av_class, opt->name, 0, &old_str))
                            break;
                        sscanf((const char *)old_str, "0x%02hhx%02hhx%02hhx%02hhx", &icol[0], &icol[1], &icol[2], &icol[3]);
                        av_freep(&old_str);
                        col[0] = icol[0] / 255.f;
                        col[1] = icol[1] / 255.f;
                        col[2] = icol[2] / 255.f;
                        col[3] = icol[3] / 255.f;
                        ImGui::SetNextItemWidth(200.f);
                        ImGui::PushID(index++);
                        ImGui::ColorEdit4(opt->name, col, ImGuiColorEditFlags_NoDragDrop);
                        ImGui::PopID();
                        icol[0] = col[0] * 255.f;
                        icol[1] = col[1] * 255.f;
                        icol[2] = col[2] * 255.f;
                        icol[3] = col[3] * 255.f;
                        snprintf(new_str, sizeof(new_str), "0x%02hhx%02hhx%02hhx%02hhx", icol[0], icol[1], icol[2], icol[3]);
                        av_opt_set(av_class, opt->name, new_str, 0);
                    }
                    break;
                case AV_OPT_TYPE_CHLAYOUT:
                    {
                        char new_layout[1024] = {0};
                        AVChannelLayout ch_layout;

                        if (av_opt_get_chlayout(av_class, opt->name, 0, &ch_layout) < 0)
                            break;
                        av_channel_layout_describe(&ch_layout, new_layout, sizeof(new_layout));
                        ImGui::SetNextItemWidth(200.f);
                        if (ImGui::InputText(opt->name, new_layout, IM_ARRAYSIZE(new_layout))) {
                            av_channel_layout_from_string(&ch_layout, new_layout);
                            av_opt_set_chlayout(av_class, opt->name, &ch_layout, 0);
                        }
                    }
                    break;
                case AV_OPT_TYPE_CONST:
                    break;
                default:
                    break;
            }

            if (ImNodes::IsNodeSelected(node->edge) && ImGui::IsItemHovered() && opt->type != AV_OPT_TYPE_CONST)
                ImGui::SetTooltip("%s", opt->help);
        }
    }
}

static int begin_group()
{
    ImGui::BeginGroup();
    return 1;
}

static void draw_filter_commands(const AVFilterContext *ctx, unsigned n, unsigned *toggle_filter,
                                 bool is_opened, bool *clean_storage, bool tree)
{
    if (1) {
        if (tree == false) {
            if (filter_nodes[n].colapsed == false && ImGui::Button("Commands"))
                filter_nodes[n].colapsed = true;
            if (filter_nodes[n].colapsed == true && ImGui::Button("Close"))
                filter_nodes[n].colapsed = false;
        }

        if ((tree == true && ImGui::TreeNode("Commands")) || ((tree == false) && filter_nodes[n].colapsed && begin_group())) {
            std::vector<OptStorage> opt_storage = filter_nodes[n].opt_storage;
            const AVOption *opt = NULL;
            unsigned opt_index = 0;

            if (is_opened && *clean_storage) {
                opt_storage.clear();
                *clean_storage = false;
            }

            while ((opt = av_opt_next(ctx->priv, opt))) {
                double min, max;

                if (!(opt->flags & AV_OPT_FLAG_RUNTIME_PARAM))
                    continue;

                if (opt->flags & AV_OPT_FLAG_READONLY)
                    continue;

                if (!query_ranges((void *)&ctx->filter->priv_class, opt, &min, &max))
                    continue;

                if (((AVOptionType)(opt->type & (~AV_OPT_TYPE_FLAG_ARRAY))) == AV_OPT_TYPE_CONST)
                    continue;

                ImGui::PushID(opt_index);
                if (opt->type & AV_OPT_TYPE_FLAG_ARRAY) {
                    AVOptionType type = (AVOptionType)(opt->type & (~AV_OPT_TYPE_FLAG_ARRAY));

                    switch (type) {
                        case AV_OPT_TYPE_FLAGS:
                        case AV_OPT_TYPE_BOOL:
                        case AV_OPT_TYPE_UINT:
                        case AV_OPT_TYPE_INT:
                        case AV_OPT_TYPE_DOUBLE:
                        case AV_OPT_TYPE_FLOAT:
                        case AV_OPT_TYPE_INT64:
                        case AV_OPT_TYPE_UINT64:
                        case AV_OPT_TYPE_STRING:
                        case AV_OPT_TYPE_COLOR:
                            if (ImGui::Button("Send")) {
                                std::stringstream arg;
                                const char separator = opt->default_val.arr->sep;

                                switch (type) {
                                    case AV_OPT_TYPE_FLAGS:
                                    case AV_OPT_TYPE_UINT:
                                        for (unsigned int i = 0; i < opt_storage[opt_index].nb_items; i++) {
                                            arg << opt_storage[opt_index].u.u32_array[i] << separator;
                                        }
                                        break;
                                    case AV_OPT_TYPE_BOOL:
                                    case AV_OPT_TYPE_INT:
                                        for (unsigned int i = 0; i < opt_storage[opt_index].nb_items; i++) {
                                            arg << opt_storage[opt_index].u.i32_array[i] << separator;
                                        }
                                        break;
                                    case AV_OPT_TYPE_INT64:
                                        for (unsigned int i = 0; i < opt_storage[opt_index].nb_items; i++) {
                                            arg << opt_storage[opt_index].u.i64_array[i] << separator;
                                        }
                                        break;
                                    case AV_OPT_TYPE_UINT64:
                                        for (unsigned int i = 0; i < opt_storage[opt_index].nb_items; i++) {
                                            arg << opt_storage[opt_index].u.u64_array[i] << separator;
                                        }
                                        break;
                                    case AV_OPT_TYPE_DOUBLE:
                                        for (unsigned int i = 0; i < opt_storage[opt_index].nb_items; i++) {
                                            arg << opt_storage[opt_index].u.dbl_array[i] << separator;
                                        }
                                        break;
                                    case AV_OPT_TYPE_FLOAT:
                                        for (unsigned int i = 0; i < opt_storage[opt_index].nb_items; i++) {
                                            arg << opt_storage[opt_index].u.flt_array[i] << separator;
                                        }
                                        break;
                                    case AV_OPT_TYPE_STRING:
                                        for (unsigned int i = 0; i < opt_storage[opt_index].nb_items; i++) {
                                            arg << opt_storage[opt_index].u.str_array[i] << separator;
                                        }
                                        break;
                                    case AV_OPT_TYPE_COLOR:
                                        for (unsigned int i = 0; i < opt_storage[opt_index].nb_items; i++) {
                                            char item[12] = {0};

                                            snprintf(item, sizeof(item)-1, "0x%02x%02x%02x%02x",
                                                     av_clip_uint8(opt_storage[opt_index].u.col_array[i].c[0] * 255),
                                                     av_clip_uint8(opt_storage[opt_index].u.col_array[i].c[1] * 255),
                                                     av_clip_uint8(opt_storage[opt_index].u.col_array[i].c[2] * 255),
                                                     av_clip_uint8(opt_storage[opt_index].u.col_array[i].c[3] * 255));

                                            arg << item << separator;
                                        }
                                        break;
                                    default:
                                        break;
                                }

                                auto x = arg.str();
                                if (!x.empty()) {
                                    x.pop_back();
                                    avfilter_graph_send_command(filter_graph, ctx->name, opt->name, x.c_str(), NULL, 0, 0);
                                }
                            }
                            ImGui::SameLine();
                            break;
                        default:
                            break;
                    }
                } else {
                    switch (opt->type) {
                        case AV_OPT_TYPE_FLAGS:
                        case AV_OPT_TYPE_BOOL:
                        case AV_OPT_TYPE_UINT:
                        case AV_OPT_TYPE_INT:
                        case AV_OPT_TYPE_DOUBLE:
                        case AV_OPT_TYPE_FLOAT:
                        case AV_OPT_TYPE_INT64:
                        case AV_OPT_TYPE_UINT64:
                        case AV_OPT_TYPE_STRING:
                        case AV_OPT_TYPE_COLOR:
                            if (ImGui::Button("Send")) {
                                char arg[1024] = { 0 };

                                switch (opt->type) {
                                    case AV_OPT_TYPE_FLAGS:
                                    case AV_OPT_TYPE_UINT:
                                        snprintf(arg, sizeof(arg) - 1, "%u", opt_storage[opt_index].u.u32);
                                        break;
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
                                    case AV_OPT_TYPE_COLOR:
                                        snprintf(arg, sizeof(arg)-1, "0x%02x%02x%02x%02x",
                                                 av_clip_uint8(opt_storage[opt_index].u.col.c[0] * 255),
                                                 av_clip_uint8(opt_storage[opt_index].u.col.c[1] * 255),
                                                 av_clip_uint8(opt_storage[opt_index].u.col.c[2] * 255),
                                                 av_clip_uint8(opt_storage[opt_index].u.col.c[3] * 255));
                                        break;
                                    default:
                                        break;
                                }

                                avfilter_graph_send_command(filter_graph, ctx->name, opt->name, arg, NULL, 0, 0);
                            }
                            ImGui::SameLine();
                            break;
                        default:
                            break;
                    }
                }

                if (opt->type & AV_OPT_TYPE_FLAG_ARRAY) {
                    AVOptionType type = (AVOptionType)(opt->type & (~AV_OPT_TYPE_FLAG_ARRAY));
                    unsigned int nb_elems = 0;

                    av_opt_get_array_size(ctx->priv, opt->name, 0, &nb_elems);
                    if (nb_elems > 0) {
                        switch (type) {
                            case AV_OPT_TYPE_FLAGS:
                            case AV_OPT_TYPE_UINT:
                                {
                                    uint32_t *value = NULL;
                                    uint32_t umax = max;
                                    uint32_t umin = min;

                                    if (opt_storage.size() <= opt_index) {
                                        OptStorage new_opt = { 0 };

                                        value = (uint32_t *)av_calloc(nb_elems, sizeof(*value));
                                        if (!value)
                                            break;

                                        new_opt.nb_items = nb_elems;
                                        new_opt.u.u32_array = value;
                                        av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, new_opt.u.u32_array);
                                        opt_storage.push_back(new_opt);
                                    } else if (nb_elems != opt_storage[opt_index].nb_items) {
                                        av_free(opt_storage[opt_index].u.u32_array);
                                        opt_storage[opt_index].u.u32_array = NULL;
                                        opt_storage[opt_index].nb_items = 0;

                                        value = (uint32_t *)av_calloc(nb_elems, sizeof(*value));
                                        if (!value)
                                            break;

                                        opt_storage[opt_index].nb_items = nb_elems;
                                        opt_storage[opt_index].u.u32_array = value;
                                        av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, opt_storage[opt_index].u.u32_array);
                                    }

                                    if (tree == false)
                                        ImGui::SetNextItemWidth(200.f);
                                    if (ImGui::DragScalarN(opt->name, ImGuiDataType_U32, opt_storage[opt_index].u.u32_array,  opt_storage[opt_index].nb_items,
                                                           1, &umin, &umax, "%u", ImGuiSliderFlags_AlwaysClamp)) {
                                        ;
                                    }
                                }
                                break;
                            case AV_OPT_TYPE_BOOL:
                            case AV_OPT_TYPE_INT:
                                {
                                    int32_t *value = NULL;
                                    int32_t imax = max;
                                    int32_t imin = min;

                                    if (opt_storage.size() <= opt_index) {
                                        OptStorage new_opt = { 0 };

                                        value = (int32_t *)av_calloc(nb_elems, sizeof(*value));
                                        if (!value)
                                            break;

                                        new_opt.nb_items = nb_elems;
                                        new_opt.u.i32_array = value;
                                        av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, new_opt.u.i32_array);
                                        opt_storage.push_back(new_opt);
                                    } else if (nb_elems != opt_storage[opt_index].nb_items) {
                                        av_free(opt_storage[opt_index].u.i32_array);
                                        opt_storage[opt_index].u.i32_array = NULL;
                                        opt_storage[opt_index].nb_items = 0;

                                        value = (int32_t *)av_calloc(nb_elems, sizeof(*value));
                                        if (!value)
                                            break;

                                        opt_storage[opt_index].nb_items = nb_elems;
                                        opt_storage[opt_index].u.i32_array = value;
                                        av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, opt_storage[opt_index].u.i32_array);
                                    }

                                    if (tree == false)
                                        ImGui::SetNextItemWidth(200.f);
                                    if (ImGui::DragScalarN(opt->name, ImGuiDataType_S32, opt_storage[opt_index].u.i32_array, opt_storage[opt_index].nb_items,
                                                           1, &imin, &imax, "%d", ImGuiSliderFlags_AlwaysClamp)) {
                                        ;
                                    }
                                }
                                break;
                            case AV_OPT_TYPE_FLOAT:
                                {
                                    float *value = NULL;
                                    float fmax = max;
                                    float fmin = min;

                                    if (opt_storage.size() <= opt_index) {
                                        OptStorage new_opt = { 0 };

                                        value = (float *)av_calloc(nb_elems, sizeof(*value));
                                        if (!value)
                                            break;

                                        new_opt.nb_items = nb_elems;
                                        new_opt.u.flt_array = value;
                                        av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, new_opt.u.flt_array);
                                        opt_storage.push_back(new_opt);
                                    } else if (nb_elems != opt_storage[opt_index].nb_items) {
                                        av_free(opt_storage[opt_index].u.flt_array);
                                        opt_storage[opt_index].u.flt_array = NULL;
                                        opt_storage[opt_index].nb_items = 0;

                                        value = (float *)av_calloc(nb_elems, sizeof(*value));
                                        if (!value)
                                            break;

                                        opt_storage[opt_index].nb_items = nb_elems;
                                        opt_storage[opt_index].u.flt_array = value;
                                        av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, opt_storage[opt_index].u.flt_array);
                                    }

                                    if (tree == false)
                                        ImGui::SetNextItemWidth(200.f);
                                    if (ImGui::DragScalarN(opt->name, ImGuiDataType_Float, opt_storage[opt_index].u.flt_array, opt_storage[opt_index].nb_items,
                                                           (fmax-fmin)/200.0, &fmin, &fmax, "%f", ImGuiSliderFlags_AlwaysClamp)) {
                                        ;
                                    }
                                }
                                break;
                            case AV_OPT_TYPE_DOUBLE:
                                {
                                    double *value = NULL;

                                    if (opt_storage.size() <= opt_index) {
                                        OptStorage new_opt = { 0 };

                                        value = (double *)av_calloc(nb_elems, sizeof(*value));
                                        if (!value)
                                            break;

                                        new_opt.nb_items = nb_elems;
                                        new_opt.u.dbl_array = value;
                                        av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, new_opt.u.dbl_array);
                                        opt_storage.push_back(new_opt);
                                    } else if (nb_elems != opt_storage[opt_index].nb_items) {
                                        av_free(opt_storage[opt_index].u.dbl_array);
                                        opt_storage[opt_index].u.dbl_array = NULL;
                                        opt_storage[opt_index].nb_items = 0;

                                        value = (double *)av_calloc(nb_elems, sizeof(*value));
                                        if (!value)
                                            break;

                                        opt_storage[opt_index].nb_items = nb_elems;
                                        opt_storage[opt_index].u.dbl_array = value;
                                        av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, opt_storage[opt_index].u.dbl_array);
                                    }

                                    if (tree == false)
                                        ImGui::SetNextItemWidth(200.f);
                                    if (ImGui::DragScalarN(opt->name, ImGuiDataType_Double, opt_storage[opt_index].u.dbl_array, opt_storage[opt_index].nb_items,
                                                           (max-min)/200.0, &min, &max, "%f", ImGuiSliderFlags_AlwaysClamp)) {
                                        ;
                                    }
                                }
                                break;
                            case AV_OPT_TYPE_STRING:
                                {
                                    char **value = NULL;

                                    if (opt_storage.size() <= opt_index) {
                                        OptStorage new_opt = { 0 };

                                        value = (char **)av_calloc(nb_elems, sizeof(*value));
                                        if (!value)
                                            break;

                                        new_opt.nb_items = nb_elems;
                                        new_opt.u.str_array = value;
                                        av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, new_opt.u.str_array);
                                        opt_storage.push_back(new_opt);
                                    } else if (nb_elems != opt_storage[opt_index].nb_items) {
                                        av_free(opt_storage[opt_index].u.str_array);
                                        opt_storage[opt_index].u.str_array = NULL;
                                        opt_storage[opt_index].nb_items = 0;

                                        value = (char **)av_calloc(nb_elems, sizeof(*value));
                                        if (!value)
                                            break;

                                        opt_storage[opt_index].nb_items = nb_elems;
                                        opt_storage[opt_index].u.str_array = value;
                                        av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, opt_storage[opt_index].u.str_array);
                                    }

                                    for (unsigned int i = 0; i < nb_elems; i++) {
                                        char label[1024] = {0};
                                        char string[1024] = { 0 };

                                        if (tree == false)
                                            ImGui::SetNextItemWidth(200.f);

                                        if (opt_storage[opt_index].u.str_array[i])
                                            memcpy(string, opt_storage[opt_index].u.str_array[i],
                                                   std::min(sizeof(string)-1, strlen(opt_storage[opt_index].u.str_array[i])));

                                        snprintf(label, sizeof(label), "%s.%u", opt->name, i);
                                        if (ImGui::InputText(label, string, IM_ARRAYSIZE(string))) {
                                            av_freep(&opt_storage[opt_index].u.str_array[i]);
                                            opt_storage[opt_index].u.str_array[i] = av_strdup(string);
                                        }
                                    }
                                }
                                break;
                            default:
                                break;
                        }
                    }
                } else {
                    switch (opt->type) {
                        case AV_OPT_TYPE_FLAGS:
                            {
                                const AVOption *copt = NULL;
                                int64_t value;
                                ImU64 imin = min;
                                ImU64 imax = max;
                                ImU64 uvalue;

                                if (av_opt_get_int(ctx->priv, opt->name, 0, &value))
                                    break;

                                if (opt_storage.size() <= opt_index) {
                                    OptStorage new_opt = { 0 };

                                    new_opt.u.u32 = (uint64_t)value;
                                    opt_storage.push_back(new_opt);
                                }

                                uvalue = value = opt_storage[opt_index].u.u32;
                                if (tree == false)
                                    ImGui::SetNextItemWidth(200.f);
                                if (ImGui::DragScalar(opt->name, ImGuiDataType_U64, &uvalue, 1, &imin, &imax, "%d", ImGuiSliderFlags_AlwaysClamp)) {
                                    value = uvalue;
                                    opt_storage[opt_index].u.u32 = uvalue;
                                }
                                if (opt->unit) {
                                    while ((copt = av_opt_next(ctx->priv, copt))) {
                                        if (!copt->unit)
                                            continue;
                                        if (strcmp(copt->unit, opt->unit) || copt->type != AV_OPT_TYPE_CONST)
                                            continue;

                                        if (tree == false)
                                            ImGui::SetNextItemWidth(200.f);
                                        ImGui::CheckboxFlags(copt->name, &uvalue, copt->default_val.i64);
                                        if (copt->help) {
                                            ImGui::SameLine();
                                            ImGui::Text("\t\t%s", copt->help);
                                        }
                                    }
                                }

                                if (value != (int64_t)uvalue) {
                                    value = uvalue;
                                    opt_storage[opt_index].u.u32 = value;
                                }
                            }
                            break;
                        case AV_OPT_TYPE_BOOL:
                            {
                                int64_t value;
                                int imin = min;
                                int imax = max;

                                if (av_opt_get_int(ctx->priv, opt->name, 0, &value))
                                    break;

                                if (opt_storage.size() <= opt_index) {
                                    OptStorage new_opt = { 0 };

                                    new_opt.u.i32 = value;
                                    opt_storage.push_back(new_opt);
                                }

                                value = opt_storage[opt_index].u.i32;
                                if (tree == false)
                                    ImGui::SetNextItemWidth(200.f);
                                if (ImGui::DragScalar(opt->name, ImGuiDataType_S32, &value, 1, &imin, &imax, "%d", ImGuiSliderFlags_AlwaysClamp)) {
                                    opt_storage[opt_index].u.i32 = value;
                                }
                            }
                            break;
                        case AV_OPT_TYPE_INT:
                            {
                                int64_t value;
                                int ivalue;
                                int imin = min;
                                int imax = max;

                                if (av_opt_get_int(ctx->priv, opt->name, 0, &value))
                                    break;

                                if (opt_storage.size() <= opt_index) {
                                    OptStorage new_opt = { 0 };

                                    new_opt.u.i32 = value;
                                    opt_storage.push_back(new_opt);
                                }

                                ivalue = opt_storage[opt_index].u.i32;
                                if (tree == false)
                                    ImGui::SetNextItemWidth(200.f);
                                if (imax < INT_MAX/2 && imin > INT_MIN/2) {
                                    if (ImGui::SliderInt(opt->name, &ivalue, imin, imax)) {
                                        opt_storage[opt_index].u.i32 = ivalue;
                                    }
                                } else {
                                    if (ImGui::DragInt(opt->name, &ivalue, imin, imax, ImGuiSliderFlags_AlwaysClamp)) {
                                        opt_storage[opt_index].u.i32 = ivalue;
                                    }
                                }
                            }
                            break;
                        case AV_OPT_TYPE_INT64:
                            {
                                int64_t value;
                                int64_t imin = min;
                                int64_t imax = max;

                                if (av_opt_get_int(ctx->priv, opt->name, 0, &value))
                                    break;

                                if (opt_storage.size() <= opt_index) {
                                    OptStorage new_opt = { 0 };

                                    new_opt.u.i64 = value;
                                    opt_storage.push_back(new_opt);
                                }
                                value = opt_storage[opt_index].u.i64;
                                if (tree == false)
                                    ImGui::SetNextItemWidth(200.f);
                                if (ImGui::DragScalar(opt->name, ImGuiDataType_S64, &value, 1, &imin, &imax, "%ld", ImGuiSliderFlags_AlwaysClamp)) {
                                    opt_storage[opt_index].u.i64 = value;
                                }
                            }
                            break;
                        case AV_OPT_TYPE_UINT64:
                            {
                                int64_t value;
                                uint64_t uvalue;
                                uint64_t umin = min;
                                uint64_t umax = max;

                                if (av_opt_get_int(ctx->priv, opt->name, 0, &value))
                                    break;

                                if (opt_storage.size() <= opt_index) {
                                    OptStorage new_opt = { 0 };

                                    new_opt.u.u64 = (uint64_t)value;
                                    opt_storage.push_back(new_opt);
                                }
                                uvalue = opt_storage[opt_index].u.u64;
                                if (tree == false)
                                    ImGui::SetNextItemWidth(200.f);
                                if (ImGui::DragScalar(opt->name, ImGuiDataType_U64, &uvalue, 1, &umin, &umax, "%lu", ImGuiSliderFlags_AlwaysClamp)) {
                                    opt_storage[opt_index].u.u64 = uvalue;
                                }
                            }
                            break;
                        case AV_OPT_TYPE_DOUBLE:
                            {
                                double value;

                                if (av_opt_get_double(ctx->priv, opt->name, 0, &value))
                                    break;

                                if (opt_storage.size() <= opt_index) {
                                    OptStorage new_opt = { 0 };

                                    new_opt.u.dbl = value;
                                    opt_storage.push_back(new_opt);
                                }
                                value = opt_storage[opt_index].u.dbl;
                                if (tree == false)
                                    ImGui::SetNextItemWidth(200.f);
                                if (ImGui::DragScalar(opt->name, ImGuiDataType_Double, &value, (max-min)/200.0, &min, &max, "%f", ImGuiSliderFlags_AlwaysClamp)) {
                                    opt_storage[opt_index].u.dbl = value;
                                }
                            }
                            break;
                        case AV_OPT_TYPE_FLOAT:
                            {
                                float fmax = max;
                                float fmin = min;
                                double value;
                                float fvalue;

                                if (av_opt_get_double(ctx->priv, opt->name, 0, &value))
                                    break;

                                if (opt_storage.size() <= opt_index) {
                                    OptStorage new_opt = { 0 };

                                    new_opt.u.flt = value;
                                    opt_storage.push_back(new_opt);
                                }
                                fvalue = opt_storage[opt_index].u.flt;
                                if (tree == false)
                                    ImGui::SetNextItemWidth(200.f);
                                if (ImGui::DragFloat(opt->name, &fvalue, (fmax - fmin)/200.f, fmin, fmax, "%f", ImGuiSliderFlags_AlwaysClamp))
                                    opt_storage[opt_index].u.flt = fvalue;
                            }
                            break;
                        case AV_OPT_TYPE_STRING:
                            {
                                char string[1024] = { 0 };
                                uint8_t *str = NULL;

                                if (opt_storage.size() <= opt_index) {
                                    OptStorage new_opt = { 0 };

                                    av_opt_get(ctx->priv, opt->name, 0, &str);
                                    new_opt.u.str = (char *)str;
                                    opt_storage.push_back(new_opt);
                                }

                                if (opt_storage[opt_index].u.str)
                                    memcpy(string, opt_storage[opt_index].u.str, std::min(sizeof(string) - 1, strlen(opt_storage[opt_index].u.str)));
                                if (tree == false)
                                    ImGui::SetNextItemWidth(200.f);
                                if (ImGui::InputText(opt->name, string, IM_ARRAYSIZE(string))) {
                                    av_freep(&opt_storage[opt_index].u.str);
                                    opt_storage[opt_index].u.str = av_strdup(string);
                                }
                            }
                            break;
                        case AV_OPT_TYPE_COLOR:
                            {
                                uint8_t icol[4];
                                uint8_t *value;
                                float col[4];

                                if (av_opt_get(ctx->priv, opt->name, 0, &value))
                                    break;

                                sscanf((char *)value, "0x%02hhX%02hhX%02hhX%02hhX", &icol[0], &icol[1], &icol[2], &icol[3]);
                                if (opt_storage.size() <= opt_index) {
                                    OptStorage new_opt = { 0 };

                                    new_opt.u.col.c[0] = icol[0] / 255.f;
                                    new_opt.u.col.c[1] = icol[1] / 255.f;
                                    new_opt.u.col.c[2] = icol[2] / 255.f;
                                    new_opt.u.col.c[3] = icol[3] / 255.f;
                                    opt_storage.push_back(new_opt);
                                }
                                col[0] = opt_storage[opt_index].u.col.c[0];
                                col[1] = opt_storage[opt_index].u.col.c[1];
                                col[2] = opt_storage[opt_index].u.col.c[2];
                                col[3] = opt_storage[opt_index].u.col.c[3];
                                if (tree == false)
                                    ImGui::SetNextItemWidth(200.f);
                                if (ImGui::ColorEdit4(opt->name, col, ImGuiColorEditFlags_NoDragDrop)) {
                                    opt_storage[opt_index].u.col.c[0] = col[0];
                                    opt_storage[opt_index].u.col.c[1] = col[1];
                                    opt_storage[opt_index].u.col.c[2] = col[2];
                                    opt_storage[opt_index].u.col.c[3] = col[3];
                                }
                            }
                            break;
                        default:
                            break;
                    }
                }

                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", opt->help);

                opt_index++;
                if (opt_index > opt_storage.size()) {
                    OptStorage new_opt = { 0 };

                    opt_storage.push_back(new_opt);
                }

                ImGui::PopID();
            }

            filter_nodes[n].opt_storage = opt_storage;

            tree ? ImGui::TreePop() : ImGui::EndGroup();
        }
    }

    if (filter_nodes[n].have_exports) {
        if (tree == false) {
            if (filter_nodes[n].show_exports == false && ImGui::Button("Exports"))
                filter_nodes[n].show_exports = true;
            if (filter_nodes[n].show_exports == true && ImGui::Button("Hide"))
                filter_nodes[n].show_exports = false;
        }

        if ((tree == true && ImGui::TreeNode("Exports")) || ((tree == false) && filter_nodes[n].show_exports && begin_group())) {
            const AVOption *opt = NULL;
            unsigned opt_index = 0;

            while ((opt = av_opt_next(ctx->priv, opt))) {
                AVOptionType type = (AVOptionType)(opt->type & (~AV_OPT_TYPE_FLAG_ARRAY));
                unsigned int nb_elems = 0;

                if (!(opt->flags & AV_OPT_FLAG_EXPORT))
                    continue;

                av_opt_get_array_size(ctx->priv, opt->name, 0, &nb_elems);

                ImGui::PushID(opt_index);
                if (tree == false)
                    ImGui::SetNextItemWidth(200.f);
                switch (type) {
                    case AV_OPT_TYPE_FLAGS:
                    case AV_OPT_TYPE_BOOL:
                    case AV_OPT_TYPE_INT:
                        {
                            int64_t value;
                            int *array;

                            if (nb_elems > 0) {
                                array = (int *)av_calloc(nb_elems, sizeof(*array));
                                if (!array)
                                    break;

                                av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, array);

                                for (unsigned i = 0; i < nb_elems; i++) {
                                    ImGui::LabelText("##export", "%s%d: %d", opt->name, i, array[i]);
                                }

                                av_freep(&array);

                                break;
                            }

                            if (av_opt_get_int(ctx->priv, opt->name, 0, &value))
                                break;

                            ImGui::LabelText("##export", "%s: %d", opt->name, (int)value);
                        }
                        break;
                    case AV_OPT_TYPE_INT64:
                        {
                            int64_t value;

                            if (av_opt_get_int(ctx->priv, opt->name, 0, &value))
                                break;

                            ImGui::LabelText("##export", "%s: %ld", opt->name, value);
                        }
                        break;
                    case AV_OPT_TYPE_UINT64:
                        {
                            int64_t value;

                            if (av_opt_get_int(ctx->priv, opt->name, 0, &value))
                                break;

                            ImGui::LabelText("##export", "%s: %lu", opt->name, (uint64_t)value);
                        }
                        break;
                    case AV_OPT_TYPE_DOUBLE:
                        {
                            double value, *array;

                            if (nb_elems > 0) {
                                array = (double *)av_calloc(nb_elems, sizeof(*array));
                                if (!array)
                                    break;

                                av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, array);

                                for (unsigned i = 0; i < nb_elems; i++) {
                                    ImGui::LabelText("##export", "%s%d: %g", opt->name, i, array[i]);
                                }

                                av_freep(&array);

                                break;
                            }

                            if (av_opt_get_double(ctx->priv, opt->name, 0, &value))
                                break;

                            ImGui::LabelText("##export", "%s: %g", opt->name, value);
                        }
                        break;
                    case AV_OPT_TYPE_FLOAT:
                        {
                            double value;
                            float *array;

                            if (nb_elems > 0) {
                                array = (float *)av_calloc(nb_elems, sizeof(*array));
                                if (!array)
                                    break;

                                av_opt_get_array(ctx->priv, opt->name, AV_OPT_ARRAY_REPLACE, 0, nb_elems, type, array);

                                for (unsigned i = 0; i < nb_elems; i++) {
                                    ImGui::LabelText("##export", "%s%d: %g", opt->name, i, array[i]);
                                }

                                av_freep(&array);

                                break;
                            }

                            if (av_opt_get_double(ctx->priv, opt->name, 0, &value))
                                break;

                            ImGui::LabelText("##export", "%s: %f", opt->name, (float)value);
                        }
                        break;
                    case AV_OPT_TYPE_STRING:
                        {
                            uint8_t *value;

                            if (av_opt_get(ctx->priv, opt->name, 0, &value))
                                break;

                            ImGui::LabelText("##export", "%s: %s", opt->name, value);

                            av_free(value);
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

            tree ? ImGui::TreePop() : ImGui::EndGroup();
        }
    }

    if (ctx->filter->flags & AVFILTER_FLAG_SUPPORT_TIMELINE) {
        if (tree ? ImGui::TreeNode("Timeline") : begin_group()) {
            ImGui::PushID(0);
            if (ImGui::Button(ctx->is_disabled ? "Enable" : "Disable"))
                *toggle_filter = n;
            ImGui::PopID();
            tree ? ImGui::TreePop() : ImGui::EndGroup();
        }
    }

    if (tree == false)
        ImGui::InvisibleButton("##inv", ImVec2(1, 1));
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
        static unsigned toggle_filter = UINT_MAX;
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
    if (node->colapsed == false && ImGui::Button("Options"))
        node->colapsed = true;

    if (node->colapsed == true && ImGui::Button("Close"))
        node->colapsed = false;

    if (node->colapsed == true) {
        for (unsigned i = 0; i < source_threads.size(); i++) {
            if (source_threads[i].joinable())
                return;
        }

        for (unsigned i = 0; i < video_sink_threads.size(); i++) {
            if (video_sink_threads[i].joinable())
                return;
        }

        for (unsigned i = 0; i < audio_sink_threads.size(); i++) {
            if (audio_sink_threads[i].joinable())
                return;
        }

        ImGui::BeginGroup();

        if (!strcmp(node->filter->name, "buffer") ||
            !strcmp(node->filter->name, "abuffer")) {
            char new_str[1024] = {0};

            ImGui::SetNextItemWidth(200.f);
            if (!node->stream_url.empty())
                memcpy(new_str, node->stream_url.c_str(), std::min(sizeof(new_str), node->stream_url.size()));
            else
                node->stream_url = std::string("<empty>");
            if (ImGui::InputText("URL", new_str, IM_ARRAYSIZE(new_str))) {
                node->stream_url.assign(new_str);
            }
        }

        draw_options(node, av_class_priv);
        ImGui::Spacing();
        draw_options(node, av_class);

        ImGui::EndGroup();
    }
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

static void set_style_colors(int style)
{
    switch (style) {
    case 0:
        ImGui::StyleColorsClassic();
        ImNodes::StyleColorsClassic();
        break;
    case 1:
        ImGui::StyleColorsDark();
        ImNodes::StyleColorsDark();
        break;
    case 2:
        ImGui::StyleColorsLight();
        ImNodes::StyleColorsLight();
        break;
    }
}

static void select_muxer(const AVOutputFormat *ofmt)
{
    recorder[0].oformat = ofmt;
}

static void handle_muxeritem(const AVOutputFormat *ofmt)
{
    if (ImGui::MenuItem(ofmt->name))
        select_muxer(ofmt);

    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", ofmt->long_name);
}

static int is_device(const AVClass *avclass)
{
    if (!avclass)
        return 0;
    return AV_IS_INPUT_DEVICE(avclass->category) || AV_IS_OUTPUT_DEVICE(avclass->category);
}

static void select_encoder(const AVCodec *codec, const bool is_audio, const int n)
{
    if (is_audio) {
        recorder[0].audio_sink_codecs[n] = codec;
    } else {
        recorder[0].video_sink_codecs[n] = codec;
    }
}

static void handle_encoderitem(const AVCodec *codec, const bool is_audio, const int n)
{
    if (ImGui::MenuItem(codec->name))
        select_encoder(codec, is_audio, n);

    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", codec->long_name);
}

static void show_filtergraph_editor(bool *p_open, bool focused)
{
    bool erased = false;
    int edge;

    if (focused)
        ImGui::SetNextWindowFocus();
    ImGui::SetNextWindowBgAlpha(editor_alpha);
    ImGui::SetNextWindowSize(ImVec2(600, 500), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("FilterGraph Editor", p_open, 0)) {
        ImGui::End();
        return;
    }

    ImNodes::EditorContextSet(node_editor_context);

    ImNodes::BeginNodeEditor();

    if (ImGui::IsKeyReleased(ImGuiKey_Enter) && ImGui::GetIO().KeyCtrl) {
        need_filters_reinit = true;
        need_muxing = false;
    }

    if (ImGui::IsKeyReleased(ImGuiKey_R) && ImGui::GetIO().KeyCtrl && need_muxing == false) {
        need_filters_reinit = true;
        need_muxing = true;
    }

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
        if (ImGui::BeginMenu("Source", filter_graph_is_valid == false)) {
            ImGui::SetTooltip("%s", "Insert Source Filter");
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
        if (ImGui::BeginMenu("Simple", filter_graph_is_valid == false)) {
            ImGui::SetTooltip("%s", "Insert Simple Filter");
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

        if (ImGui::BeginMenu("Complex", filter_graph_is_valid == false)) {
            ImGui::SetTooltip("%s", "Insert Complex Filter");
            if (ImGui::BeginMenu("Video")) {
                static ImGuiTextFilter imgui_filter;
                const AVFilter *filter = NULL;
                void *iterator = NULL;

                ImGui::SetTooltip("%s", "Complex Video Filters");
                imgui_filter.Draw();
                while ((filter = av_filter_iterate(&iterator))) {
                    if (!is_complex_video_filter(filter))
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

                ImGui::SetTooltip("%s", "Complex Audio Filters");
                imgui_filter.Draw();
                while ((filter = av_filter_iterate(&iterator))) {
                    if (!is_complex_audio_filter(filter))
                        continue;

                    if (!imgui_filter.PassFilter(filter->name))
                        continue;

                    handle_nodeitem(filter, click_pos);
                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Media", filter_graph_is_valid == false)) {
            const AVFilter *filter = NULL;
            void *iterator = NULL;

            ImGui::SetTooltip("%s", "Insert Media Filter");
            while ((filter = av_filter_iterate(&iterator))) {
                if (!is_media_filter(filter))
                    continue;

                handle_nodeitem(filter, click_pos);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Sink", filter_graph_is_valid == false)) {
            ImGui::SetTooltip("%s", "Insert Sink Filter");
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
                filter_graph_nb_threads = std::max(filter_graph_nb_threads, 0);
                ImGui::InputInt("Auto Conversion Type for FilterGraph", &filter_graph_auto_convert_flags);
                filter_graph_auto_convert_flags = std::clamp(filter_graph_auto_convert_flags, -1, 0);
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

                ImGui::DragFloat2("OSD Fullscreen Position", osd_fullscreen_pos, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                ImGui::DragFloat("OSD Fullscreen Alpha", &osd_alpha, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
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
                const char *format_items[] = { "Float", "Double" };
                const bool format_values[] = { 0, 1 };

                if (ImGui::BeginCombo("Audio Sample Format", format_items[audio_format], 0)) {
                    for (int n = 0; n < IM_ARRAYSIZE(format_items); n++) {
                        const bool is_selected = format_values[n] == audio_format;

                        if (ImGui::Selectable(format_items[n], is_selected))
                            audio_format = format_values[n];

                        if (is_selected)
                            ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                ImGui::DragInt("Audio Samples Queue Size", &audio_queue_size, 1.f, 4, 256, "%d", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
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
                const char *depth_items[] = { "8-bit", "16-bit" };
                const unsigned depth_values[] = { 0, 1 };

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

                if (ImGui::BeginCombo("Depth mode", depth_items[depth], 0)) {
                    for (int n = 0; n < IM_ARRAYSIZE(depth_values); n++) {
                        const bool is_selected = depth == depth_values[n];

                        if (ImGui::Selectable(depth_items[n], is_selected))
                            depth = depth_values[n];

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
                    style_colors = 0;
                }
                if (ImGui::MenuItem("Dark")) {
                    style_colors = 1;
                }
                if (ImGui::MenuItem("Light")) {
                    style_colors = 2;
                }
                set_style_colors(style_colors);
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Widgets")) {
                if (ImGui::BeginMenu("Background Alpha")) {
                    ImGui::DragFloat("Commands", &commands_alpha, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                    ImGui::DragFloat("Console", &console_alpha, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                    ImGui::DragFloat("Dump", &dump_alpha, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                    ImGui::DragFloat("Editor", &editor_alpha, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                    ImGui::DragFloat("Help", &help_alpha, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                    ImGui::DragFloat("Info", &info_alpha, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                    ImGui::DragFloat("Log", &log_alpha, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                    ImGui::DragFloat("Record", &record_alpha, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                    ImGui::DragFloat("Sink", &sink_alpha, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                    ImGui::DragFloat("Version", &version_alpha, 0.01f, 0.f, 1.f, "%f", ImGuiSliderFlags_AlwaysClamp | ImGuiSliderFlags_NoInput);
                    ImGui::EndMenu();
                }
                ImGui::EndMenu();
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Graph")) {
            if (ImGui::BeginMenu("Export", filter_graph_is_valid == true)) {
                ImGui::SetTooltip("%s", "Export FilterGraph");
                if (ImGui::BeginMenu("Save as Script")) {
                    static char file_name[1024] = { 0 };

                    ImGui::InputText("File name:", file_name, IM_ARRAYSIZE(file_name));
                    if (strlen(file_name) > 0 && ImGui::Button("Save")) {
                        exportfile_filter_graph(file_name);
                        memset(file_name, 0, sizeof(file_name));
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Import", filter_graph_is_valid == false &&
                                 filter_links.size() == 0 &&
                                 filter_nodes.size() == 0)) {
                ImGui::SetTooltip("%s", "Import FilterGraph");
                if (ImGui::BeginMenu("Load Script")) {
                    static char file_name[1024] = { 0 };

                    ImGui::InputText("File name:", file_name, IM_ARRAYSIZE(file_name));
                    if (strlen(file_name) > 0 && ImGui::Button("Load")) {
                        av_freep(&import_script_file_name);
                        import_script_file_name = av_asprintf("%s", file_name);
                        memset(file_name, 0, sizeof(file_name));
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Record", filter_graph_is_valid == true)) {
                static char file_name[1024] = { 0 };

                recorder.resize(1);

                ImGui::SetTooltip("%s", "Record FilterGraph Output");
                ImGui::Text("File: %s", recorder[0].filename ? recorder[0].filename : "<none>");
                ImGui::Text("Format: %s", recorder[0].oformat ? recorder[0].oformat->name : "<none>");

                recorder[0].audio_sink_codecs.resize(abuffer_sinks.size());
                recorder[0].video_sink_codecs.resize(buffer_sinks.size());

                for (unsigned i = 0; i < recorder[0].audio_sink_codecs.size(); i++) {
                    ImGui::Text("Audio Encoder.%u: %s", i, recorder[0].audio_sink_codecs[i] ? recorder[0].audio_sink_codecs[i]->name : "<none>");
                }

                for (unsigned i = 0; i < recorder[0].video_sink_codecs.size(); i++) {
                    ImGui::Text("Video Encoder.%u: %s", i, recorder[0].video_sink_codecs[i] ? recorder[0].video_sink_codecs[i]->name : "<none>");
                }

                ImGui::Separator();

                for (unsigned i = 0; i < recorder[0].audio_sink_codecs.size(); i++) {
                    char menu_name[1024] = { 0 };

                    snprintf(menu_name, sizeof(menu_name), "Audio Encoder.%u", i);
                    if (ImGui::BeginMenu(menu_name)) {
                        const AVCodec *ocodec;
                        void *iterator = NULL;

                        ImGui::SetTooltip("Audio Stream %d Encoder", i);
                        while ((ocodec = av_codec_iterate(&iterator))) {
                            if (!av_codec_is_encoder(ocodec) ||
                                ocodec->type != AVMEDIA_TYPE_AUDIO)
                                continue;

                            handle_encoderitem(ocodec, true, i);
                        }
                        ImGui::EndMenu();
                    }
                }

                for (unsigned i = 0; i < recorder[0].video_sink_codecs.size(); i++) {
                    char menu_name[1024] = { 0 };

                    snprintf(menu_name, sizeof(menu_name), "Video Encoder.%u", i);
                    if (ImGui::BeginMenu(menu_name)) {
                        const AVCodec *ocodec;
                        void *iterator = NULL;

                        ImGui::SetTooltip("Video Stream %d Encoder", i);
                        while ((ocodec = av_codec_iterate(&iterator))) {
                            if (!av_codec_is_encoder(ocodec) ||
                                ocodec->type != AVMEDIA_TYPE_VIDEO)
                                continue;

                            handle_encoderitem(ocodec, false, i);
                        }
                        ImGui::EndMenu();
                    }
                }

                if (ImGui::BeginMenu("Format")) {
                    const char *last_name = NULL;
                    const AVOutputFormat *ofmt;
                    void *iterator = NULL;

                    ImGui::SetTooltip("%s", "Output Formats");
                    while ((ofmt = av_muxer_iterate(&iterator))) {
                        if (is_device(ofmt->priv_class))
                            continue;

                        if (!last_name || strcmp(last_name, ofmt->name)) {
                            handle_muxeritem(ofmt);
                            last_name = ofmt->name;
                        }
                    }
                    ImGui::EndMenu();
                }

                ImGui::InputText("File name:", file_name, IM_ARRAYSIZE(file_name));
                if (strlen(file_name) > 0 && ImGui::Button("Set")) {
                    av_freep(&recorder[0].filename);
                    recorder[0].filename = av_strdup(file_name);
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
        bool disabled = false;

        edge = filter_node->edge;
        edge2pad[edge] = (Edge2Pad { i, false, false, 0, AVMEDIA_TYPE_UNKNOWN });
        if (!ImNodes::IsNodeSelected(edge))
            disabled = true;
        if (disabled)
            ImGui::BeginDisabled();
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
            ImGui::SetTooltip("%s : %s", filter_node->filter_label, filter_node->filter->description);
        ImNodes::EndNodeTitleBar();
        ImNodes::BeginStaticAttribute(edge);
        draw_node_options(filter_node);
        ImNodes::EndStaticAttribute();
        if (!filter_node->probe) {
            ImNodes::EndNode();
            if (disabled)
                ImGui::EndDisabled();
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
            } else if (media_type == AVMEDIA_TYPE_AUDIO) {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(255, 255,   0, 255));
            } else {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(255,   0,   0, 255));
            }
            edge2pad[edge] = (Edge2Pad { i, false, false, j, media_type });
            filter_node->inpad_edges[j] = edge;
            ImNodes::BeginInputAttribute(edge);
            ImGui::TextUnformatted(avfilter_pad_get_name(filter_ctx->input_pads, j));
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
            } else if (media_type == AVMEDIA_TYPE_AUDIO) {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(255, 255,   0, 255));
            } else {
                ImNodes::PushColorStyle(ImNodesCol_Pin, IM_COL32(255,   0,   0, 255));
            }
            edge2pad[edge] = (Edge2Pad { i, false, true, j, media_type });
            filter_node->outpad_edges[j] = edge;
            ImNodes::BeginOutputAttribute(edge);
            ImGui::TextUnformatted(avfilter_pad_get_name(filter_ctx->output_pads, j));
            ImNodes::EndOutputAttribute();
            ImNodes::PopColorStyle();
        }

        ImNodes::EndNode();
        if (disabled)
            ImGui::EndDisabled();
        ImNodes::SetNodeDraggable(filter_node->edge, true);
    }

    for (unsigned i = 0; i < filter_links.size();) {
        const std::pair<int, int> p = filter_links[i];

        if (edge2pad[p.first].removed  == true ||
            edge2pad[p.second].removed == true) {
            edge2pad[p.first].removed    = true;
            edge2pad[p.second].removed   = true;
            edge2pad[p.first].is_output  = false;
            edge2pad[p.second].is_output = false;

            filter_links.erase(filter_links.begin() + i);
        } else if (edge2pad[p.first].type  == AVMEDIA_TYPE_UNKNOWN ||
            edge2pad[p.second].type == AVMEDIA_TYPE_UNKNOWN) {
            edge2pad[p.first].removed    = true;
            edge2pad[p.second].removed   = true;
            edge2pad[p.first].is_output  = false;
            edge2pad[p.second].is_output = false;

            filter_links.erase(filter_links.begin() + i);
        } else if (edge2pad[p.first].is_output == edge2pad[p.second].is_output) {
            edge2pad[p.first].removed    = true;
            edge2pad[p.second].removed   = true;
            edge2pad[p.first].is_output  = false;
            edge2pad[p.second].is_output = false;

            filter_links.erase(filter_links.begin() + i);
        } else {
            i++;
        }
    }

    for (unsigned i = 0; i < filter_links.size(); i++) {
        const std::pair<int, int> p = filter_links[i];

        ImNodes::Link(i, p.first, p.second);
    }

    if (show_mini_map == true)
        ImNodes::MiniMap(0.2f, mini_map_location);
    ImNodes::EndNodeEditor();

    for (unsigned i = 0; i < source_threads.size(); i++) {
        if (source_threads[i].joinable()) {
            ImGui::End();
            return;
        }
    }

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
            FilterNode copy = {0};

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
            FilterNode node = {0};

            node.filter = type == AVMEDIA_TYPE_AUDIO ? abuffersink : buffersink;
            node.id = filter_nodes.size();
            node.filter_name = av_strdup(node.filter->name);
            node.filter_label = av_asprintf("%s@%d", node.filter->name, node.id);
            node.filter_options = NULL;
            node.ctx_options = NULL;
            node.probe = NULL;
            node.ctx = NULL;
            node.pos = find_node_spot(src.pos);
            node.colapsed = false;
            node.have_exports = false;
            node.show_exports = false;
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
    static ImGuiTextFilter imgui_filter;

    ImGui::BeginGroup();

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

        if (!imgui_filter.PassFilter(ctx->name))
            continue;

        if (ImGui::Selectable(ctx->name, is_selected)) {
            selected_filter = n;
        }

        if (ImGui::IsItemActive() || ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", ctx->filter->description);
        }

        if (ImGui::IsItemClicked() && ImGui::IsItemActive()) {
            selected_filter = n;
            is_opened = true;
        }

        if (is_opened && selected_filter == n)
            draw_filter_commands(ctx, n, toggle_filter, is_opened, &clean_storage, true);
    }

    ImGui::EndGroup();
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
    ImGui::SetNextWindowBgAlpha(commands_alpha);
    ImGui::SetNextWindowSize(ImVec2(500, 200), ImGuiCond_FirstUseEver);
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
    ImGui::SetNextWindowBgAlpha(dump_alpha);
    if (!ImGui::Begin("FilterGraph Dump", p_open, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        return;
    }
    ImGui::TextUnformatted(graphdump_text);
    ImGui::End();
}

static void log_callback(void *ptr, int level, const char *fmt, va_list args)
{
    log_mutex.lock();

    if (log_level >= level) {
        int new_size;

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
        new_size = log_buffer.size();
        log_lines_levels.push_back(level);
        log_lines_offsets.push_back(new_size);
    }

    log_mutex.unlock();
}

static void show_log(bool *p_open, bool focused)
{
    static ImGuiTextFilter filter;

    if (focused)
        ImGui::SetNextWindowFocus();
    ImGui::SetNextWindowSize(ImVec2(500, 100), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(log_alpha);
    if (!ImGui::Begin("FilterGraph Log", p_open, 0)) {
        ImGui::End();
        return;
    }

    filter.Draw("###Log Filter", ImGui::GetWindowSize().x);
    for (int line_no = 0; line_no < log_lines_offsets.Size; line_no++) {
        const char *line_start = log_buffer.begin() + ((line_no > 1) ? log_lines_offsets[line_no-1] : 0);
        const char *line_end = log_buffer.begin() + log_lines_offsets[line_no];
        ImVec4 color;

        if (!filter.IsActive() || filter.PassFilter(line_start, line_end)) {
            const int line_length = line_end - line_start;
            const int level = log_lines_levels[line_no];

            switch (level) {
                case AV_LOG_PANIC:
                    color = ImVec4(1.f, 0.3f, 1.f, 1.f);
                    break;
                case AV_LOG_FATAL:
                    color = ImVec4(1.f, 0, 1.f, 1.f);
                    break;
                case AV_LOG_ERROR:
                    color = ImVec4(1.f, 0.f, 0.f, 1.f);
                    break;
                case AV_LOG_WARNING:
                    color = ImVec4(1.f, 1.f, 0, 1.f);
                    break;
                case AV_LOG_INFO:
                    color = ImVec4(0, 0, 1.f, 1.f);
                    break;
                case AV_LOG_VERBOSE:
                    color = ImVec4(0, 0.5f, 1.f, 1.f);
                    break;
                case AV_LOG_DEBUG:
                    color = ImVec4(0.5f, 0.5f, 1.f, 1.f);
                    break;
                default:
                    color = ImVec4(1.f, 1.f, 1.f, 1.f);
                    break;
            }

            ImGui::TextColored(color, "%.*s", line_length, line_start);
        }
    }

    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);

    ImGui::End();
}

static void show_record(bool *p_open, bool focused)
{
    if (focused)
        ImGui::SetNextWindowFocus();
    ImGui::SetNextWindowBgAlpha(record_alpha);
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("FilterGraph Record", p_open, 0)) {
        ImGui::End();
        return;
    }

    if (recorder.size() > 0) {
        for (unsigned i = 0; i < recorder[0].ostreams.size(); i++) {
            OutputStream *os = &recorder[0].ostreams[i];

            if (os->last_codec)
                ImGui::Text("Encoder.%d: %s", i, os->last_codec->name);
            ImGui::Separator();
            ImGui::Text("Last Frame Filtering Time: %g", (os->end_flt_time - os->start_flt_time)/ 1000000.);
            ImGui::Text("Overall Filtering Time: %g", os->elapsed_flt_time / 1000000.);
            ImGui::Separator();
            ImGui::Text("Last Frame Encoding Time: %g", (os->end_enc_time - os->start_enc_time)/ 1000000.);
            ImGui::Text("Overall Encoding Time: %g", os->elapsed_enc_time / 1000000.);
            ImGui::Separator();
            ImGui::Text("Last Frame Size: %lu", os->last_frame_size);
            ImGui::Text("Last Packet Size: %lu", os->last_packet_size);
            ImGui::Text("Last Frame Compression Ratio: %g", os->last_packet_size / (double)os->last_frame_size);
            ImGui::Separator();
            ImGui::Text("Frame Size Sum: %lu", os->sum_of_frames);
            ImGui::Text("Packet Size Sum: %lu", os->sum_of_packets);
            ImGui::Text("Compression Ratio : %g", os->sum_of_packets / (double)os->sum_of_frames);
            ImGui::Separator();
            ImGui::Separator();
        }
    }

    ImGui::End();
}

int main(int, char**)
{
    ALCint attribs[] = { ALC_FREQUENCY, output_sample_rate, 0, 0 };

    avdevice_register_all();

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
    float highDPIscaleFactor = 1.f;
    float xscale, yscale;
    glfwGetMonitorContentScale(glfwGetPrimaryMonitor(), &xscale, &yscale);
    if (xscale > 1.f || yscale > 1.f) {
        highDPIscaleFactor = xscale;
        glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    }
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
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
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
    io.KeyRepeatDelay = 0.5f;

    // Setup Dear ImGui style
    set_style_colors(style_colors);
    if (highDPIscaleFactor > 1.f) {
        ImGuiStyle &style = ImGui::GetStyle();
        style.ScaleAllSizes(highDPIscaleFactor);
    }

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

        if (filter_graph_is_valid == false) {
            kill_recorder_threads();
            kill_source_threads();
            source_threads.clear();
        }

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
            importfile_filter_graph(import_script_file_name);
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

        if (filter_graph_is_valid && (recorder.size() == 0 || recorder[0].format_ctx == NULL)) {
            for (unsigned i = 0; i < buffer_sinks.size(); i++) {
                BufferSink *sink = &buffer_sinks[i];
                ring_item_t item = { NULL, 0 };

                if (paused && !framestep)
                    continue;

                if (sink->qpts > min_qpts)
                    continue;

                if (ring_buffer_is_empty(&sink->render_frames)) {
                    if (!ring_buffer_is_empty(&sink->empty_frames))
                        notify_worker(sink, &mutexes[i], &cv[i]);
                    continue;
                }

                if (ring_buffer_is_full(&sink->empty_frames))
                    continue;

                ring_buffer_dequeue(&sink->render_frames, &item);
                if (!item.frame)
                    continue;
                av_frame_unref(item.frame);
                ring_buffer_enqueue(&sink->empty_frames, item);
                notify_worker(sink, &mutexes[i], &cv[i]);
            }
        }

        if (filter_graph_is_valid && (recorder.size() == 0 || recorder[0].format_ctx == NULL)) {
            for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
                BufferSink *sink = &abuffer_sinks[i];
                ALint processed = 0;

                if (ring_buffer_is_full(&sink->empty_frames))
                    continue;

                alGetSourcei(sink->source, AL_BUFFERS_PROCESSED, &processed);
                if (processed <= 0)
                    continue;

                while (processed-- > 0) {
                    ring_item_t item = { NULL, 0 };

                    ring_buffer_dequeue(&sink->render_frames, &item);
                    if (!item.frame)
                        break;
                    alSourceUnqueueBuffers(sink->source, 1, &item.id.u.a);
                    av_frame_unref(item.frame);
                    ring_buffer_enqueue(&sink->empty_frames, item);
                    notify_worker(sink, &amutexes[i], &acv[i]);
                }
            }
        }

        if (filter_graph_is_valid && (recorder.size() == 0 || recorder[0].format_ctx == NULL)) {
            for (unsigned i = 0; i < buffer_sinks.size(); i++) {
                BufferSink *sink = &buffer_sinks[i];
                ring_item_t item = { NULL, 0 };

                if (ring_buffer_is_empty(&sink->consume_frames)) {
                    if (!ring_buffer_is_empty(&sink->empty_frames))
                        notify_worker(sink, &mutexes[i], &cv[i]);
                    continue;
                }

                if (!ring_buffer_is_empty(&sink->render_frames))
                    continue;

                ring_buffer_dequeue(&sink->consume_frames, &item);
                if (!item.frame)
                    continue;
                ring_buffer_enqueue(&sink->render_frames, item);
            }
        } else {
            focus_buffersink_window = UINT_MAX;
            last_buffersink_window = 0;
        }

        if (filter_graph_is_valid && (recorder.size() == 0 || recorder[0].format_ctx == NULL)) {
            for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
                BufferSink *sink = &abuffer_sinks[i];
                ALint queued = 0;

                alGetSourcei(sink->source, AL_BUFFERS_QUEUED, &queued);
                while (queued < sink->audio_queue_size) {
                    ring_item_t item = { NULL, 0 };

                    if (ring_buffer_is_empty(&sink->consume_frames)) {
                        if (!ring_buffer_is_empty(&sink->empty_frames))
                            notify_worker(sink, &amutexes[i], &acv[i]);
                        break;
                    }

                    if (ring_buffer_is_full(&sink->render_frames))
                        break;

                    ring_buffer_dequeue(&sink->consume_frames, &item);
                    if (!item.frame)
                        break;
                    ring_buffer_enqueue(&sink->render_frames, item);
                    queue_sound(sink, item);
                    queued++;
                }
            }
        } else {
            focus_abuffersink_window = UINT_MAX;
            last_abuffersink_window = 0;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (recorder.size() == 0 || recorder[0].format_ctx == NULL) {
            if (filter_graph_is_valid && show_buffersink_window == true) {
                for (unsigned i = 0; i < buffer_sinks.size(); i++) {
                    BufferSink *sink = &buffer_sinks[i];
                    ring_item_t item = { NULL, 0 };

                    ring_buffer_peek(&sink->render_frames, &item, 0);
                    draw_frame(&show_buffersink_window, item, sink);
                }
            }

            if (filter_graph_is_valid && show_abuffersink_window == true) {
                for (unsigned i = 0; i < abuffer_sinks.size(); i++) {
                    BufferSink *sink = &abuffer_sinks[i];
                    ring_item_t item = { NULL, 0 };

                    ring_buffer_peek(&sink->render_frames, &item, 0);
                    draw_aframe(&show_abuffersink_window, item, sink);
                }
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
        focused = ImGui::IsKeyReleased(ImGuiKey_F6);
        if (focused)
            show_record_window = true;
        if (show_record_window)
            show_record(&show_record_window, focused);
        focused = ImGui::IsKeyReleased(ImGuiKey_F2);
        if (focused)
            show_filtergraph_editor_window = true;
        if (show_filtergraph_editor_window)
            show_filtergraph_editor(&show_filtergraph_editor_window, focused);
        show_help = ImGui::IsKeyDown(ImGuiKey_F1);
        if (show_help)
            draw_help(&show_help);
        show_version = ImGui::IsKeyDown(ImGuiKey_F12);
        if (show_version)
            draw_version(&show_version);
        show_info = ImGui::IsKeyDown(ImGuiKey_I) && !io.WantTextInput;
        if (show_info)
            draw_info(&show_info, ImGui::GetIO().KeyShift);
        show_console ^= ImGui::IsKeyReleased(ImGuiKey_Escape);
        if (show_console)
            draw_console(&show_console);
        if (filter_graph_is_valid && show_abuffersink_window == true && abuffer_sinks.size() > 0 &&
            ImGui::GetIO().KeyAlt && ImGui::IsKeyPressed(ImGuiKey_Tab, false) &&
            last_abuffersink_window < UINT_MAX && focus_abuffersink_window == UINT_MAX) {
            focus_abuffersink_window = (last_abuffersink_window + 1) % abuffer_sinks.size();
            last_abuffersink_window = UINT_MAX;
        }
        if (filter_graph_is_valid && show_buffersink_window == true && buffer_sinks.size() > 0 &&
            ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Tab, false) &&
            last_buffersink_window < UINT_MAX && focus_buffersink_window == UINT_MAX) {
            focus_buffersink_window = (last_buffersink_window + 1) % buffer_sinks.size();
            last_buffersink_window = UINT_MAX;
        }

        // Rendering
        ImGui::Render();
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        if (restart_display == true)
            break;
    }

    need_filters_reinit = true;

    kill_recorder_threads();
    recorder.clear();

    if (play_sound_thread.joinable())
        play_sound_thread.join();
    play_sources.clear();

    kill_source_threads();
    kill_audio_sink_threads();
    kill_video_sink_threads();

    source_threads.clear();
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
