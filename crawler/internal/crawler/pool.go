// Package crawler provides the worker pool implementation.
package crawler

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// Pool manages a pool of crawler workers with rate limiting.
type Pool struct {
	workers   int
	delayMs   int
	queue     chan string
	wg        sync.WaitGroup
	ctx       context.Context
	cancel    context.CancelFunc
	active    int64
	processed int64
	crawler   *Crawler
	started   bool
	mu        sync.Mutex

	// Rate limiting
	lastRequest time.Time
	rateMu      sync.Mutex
}

// newPool creates a new worker pool.
func newPool(workers, delayMs int) *Pool {
	return &Pool{
		workers: workers,
		delayMs: delayMs,
		queue:   make(chan string, workers*100), // Buffer for burst capacity
	}
}

// Start initializes and starts all workers.
func (p *Pool) Start(ctx context.Context) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.started {
		return
	}

	p.ctx, p.cancel = context.WithCancel(ctx)
	p.started = true

	// Start workers
	for i := 0; i < p.workers; i++ {
		p.wg.Add(1)
		go p.worker(i)
	}
}

// worker is the main worker loop that processes URLs from the queue.
func (p *Pool) worker(id int) {
	defer p.wg.Done()

	for {
		select {
		case <-p.ctx.Done():
			return
		case url, ok := <-p.queue:
			if !ok {
				return
			}

			atomic.AddInt64(&p.active, 1)

			// Rate limit before processing
			p.rateLimit()

			// Process the URL
			if p.crawler != nil {
				err := p.crawler.Process(p.ctx, url, p.Enqueue)
				if err != nil {
					// Log error but continue crawling
					// In production, use structured logging
				}
			}

			atomic.AddInt64(&p.active, -1)
			atomic.AddInt64(&p.processed, 1)
		}
	}
}

// rateLimit enforces minimum delay between requests to the same domain.
func (p *Pool) rateLimit() {
	if p.delayMs <= 0 {
		return
	}

	p.rateMu.Lock()
	defer p.rateMu.Unlock()

	elapsed := time.Since(p.lastRequest)
	required := time.Duration(p.delayMs) * time.Millisecond

	if elapsed < required && !p.lastRequest.IsZero() {
		time.Sleep(required - elapsed)
	}

	p.lastRequest = time.Now()
}

// Enqueue adds a URL to the queue for processing.
// Non-blocking - if queue is full, URL is dropped to prevent unbounded growth.
func (p *Pool) Enqueue(url string) {
	select {
	case <-p.ctx.Done():
		return
	default:
	}

	select {
	case p.queue <- url:
		// Successfully enqueued
	default:
		// Queue is full, skip to prevent blocking
		// This is acceptable as the queue is large and this is rare
	}
}

// Wait blocks until the queue is empty and all workers finish.
func (p *Pool) Wait() {
	// Poll for completion with timeout
	timeout := time.NewTimer(10 * time.Minute)
	defer timeout.Stop()

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout.C:
			// Force shutdown on timeout
			p.Stop()
			return
		case <-ticker.C:
			// Check if queue is empty and no active workers
			queueLen := len(p.queue)
			activeCount := atomic.LoadInt64(&p.active)

			if queueLen == 0 && activeCount == 0 {
				// Double-check after a short delay to avoid race condition
				time.Sleep(50 * time.Millisecond)
				if len(p.queue) == 0 && atomic.LoadInt64(&p.active) == 0 {
					p.Stop()
					return
				}
			}
		}
	}
}

// Stop gracefully shuts down the pool.
func (p *Pool) Stop() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.started {
		return
	}

	// Cancel context to signal workers to stop
	if p.cancel != nil {
		p.cancel()
	}

	// Close the queue so workers can exit
	close(p.queue)

	// Wait for workers with timeout
	done := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Workers finished cleanly
	case <-time.After(30 * time.Second):
		// Timeout waiting for workers - they will be force-killed
	}

	p.started = false
}

// Stats returns current pool statistics.
func (p *Pool) Stats() (queued, active, processed int64) {
	return int64(len(p.queue)), atomic.LoadInt64(&p.active), atomic.LoadInt64(&p.processed)
}

// QueueLen returns the current queue length.
func (p *Pool) QueueLen() int {
	return len(p.queue)
}

// ActiveCount returns the number of active workers.
func (p *Pool) ActiveCount() int64 {
	return atomic.LoadInt64(&p.active)
}

// ProcessedCount returns the total number of processed URLs.
func (p *Pool) ProcessedCount() int64 {
	return atomic.LoadInt64(&p.processed)
}
