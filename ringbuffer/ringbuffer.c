#include <stdint.h>
#include <libavutil/mem.h>
#include <libavutil/frame.h>

typedef struct ring_buffer_t {
    unsigned size;
    unsigned mask;
    AVFrame **buffer;
    unsigned tail_index;
    unsigned head_index;
} ring_buffer_t;

static int ring_buffer_init(ring_buffer_t *buffer, unsigned size)
{
    buffer->tail_index = 0;
    buffer->head_index = 0;
    buffer->size = 1U << av_ceil_log2(size);
    buffer->mask = buffer->size-1U;

    buffer->buffer = (AVFrame **)av_calloc(buffer->size, sizeof(*buffer->buffer));
    if (!buffer->buffer) {
        buffer->size = 0;
        buffer->mask = 0;
        return AVERROR(ENOMEM);
    }
    return 0;
}

static void ring_buffer_free(ring_buffer_t *buffer)
{
    av_freep(&buffer->buffer);
    buffer->mask = 0;
    buffer->size = 0;
}

static inline int ring_buffer_is_empty(ring_buffer_t *buffer)
{
    return buffer->head_index == buffer->tail_index;
}

static inline int ring_buffer_is_full(ring_buffer_t *buffer)
{
    return ((buffer->head_index - buffer->tail_index) & buffer->mask) == buffer->mask;
}

static void ring_buffer_enqueue(ring_buffer_t *buffer, AVFrame *data)
{
    buffer->buffer[buffer->head_index] = data;
    buffer->head_index = ((buffer->head_index + 1U) & buffer->mask);
}

static void ring_buffer_dequeue(ring_buffer_t *buffer, AVFrame **data)
{
    if (!ring_buffer_is_empty(buffer)) {
        *data = buffer->buffer[buffer->tail_index];
        buffer->buffer[buffer->tail_index] = NULL;
        buffer->tail_index = ((buffer->tail_index + 1U) & buffer->mask);
    }
}

static void ring_buffer_peek(ring_buffer_t *buffer, AVFrame **data, unsigned index)
{
    unsigned max = (buffer->head_index - buffer->tail_index) & buffer->mask;
    if (max > 0) {
        unsigned data_index = (buffer->tail_index + std::min(index, max - 1)) & buffer->mask;
        *data = buffer->buffer[data_index];
    }
}

static inline unsigned ring_buffer_num_items(ring_buffer_t *buffer)
{
    unsigned ret;

    ret = (buffer->head_index - buffer->tail_index) & buffer->mask;

    return ret;
}
