#include <stdint.h>
#include <libavutil/frame.h>

#define RING_BUFFER_SIZE 8U

#define RING_BUFFER_MASK (RING_BUFFER_SIZE-1)

typedef struct ring_buffer_t {
    AVFrame *buffer[RING_BUFFER_SIZE];
    unsigned tail_index;
    unsigned head_index;
} ring_buffer_t;

static void ring_buffer_init(ring_buffer_t *buffer)
{
    buffer->tail_index = 0;
    buffer->head_index = 0;
}

static inline int ring_buffer_is_empty(ring_buffer_t *buffer)
{
    return buffer->head_index == buffer->tail_index;
}

static inline int ring_buffer_is_full(ring_buffer_t *buffer)
{
    return ((buffer->head_index - buffer->tail_index) & RING_BUFFER_MASK) == RING_BUFFER_MASK;
}

static void ring_buffer_enqueue(ring_buffer_t *buffer, AVFrame *data, std::mutex *mutex)
{
    mutex->lock();
    buffer->buffer[buffer->head_index] = data;
    buffer->head_index = ((buffer->head_index + 1U) & RING_BUFFER_MASK);
    mutex->unlock();
}

static void ring_buffer_dequeue(ring_buffer_t *buffer, AVFrame **data, std::mutex *mutex)
{
    mutex->lock();
    if (!ring_buffer_is_empty(buffer)) {
        *data = buffer->buffer[buffer->tail_index];
        buffer->buffer[buffer->tail_index] = NULL;
        buffer->tail_index = ((buffer->tail_index + 1U) & RING_BUFFER_MASK);
    }
    mutex->unlock();
}

static void ring_buffer_peek(ring_buffer_t *buffer, AVFrame **data, unsigned index, std::mutex *mutex)
{
    mutex->lock();
    unsigned max = (buffer->head_index - buffer->tail_index) & RING_BUFFER_MASK;
    if (max > 0) {
        unsigned data_index = (buffer->tail_index + std::min(index, max - 1)) & RING_BUFFER_MASK;
        *data = buffer->buffer[data_index];
    }
    mutex->unlock();
}

static inline unsigned ring_buffer_num_items(ring_buffer_t *buffer, std::mutex *mutex)
{
    unsigned ret;

    mutex->lock();
    ret = (buffer->head_index - buffer->tail_index) & RING_BUFFER_MASK;
    mutex->unlock();

    return ret;
}
