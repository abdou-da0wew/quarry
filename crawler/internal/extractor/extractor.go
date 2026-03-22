// Package extractor provides HTML content extraction using goquery.
// Extracts title, body text, and links from HTML documents.
package extractor

import (
	"net/url"
	"regexp"
	"strings"

	"github.com/PuerkitoBio/goquery"
)

// Page represents extracted content from a web page.
type Page struct {
	URL         string
	Title       string
	Body        string
	Links       []string
	ContentType string
}

// Extractor extracts structured content from HTML.
type Extractor struct {
	// BaseDomain is the domain to limit crawling to.
	BaseDomain string
	// MaxBodySize limits the body text size in bytes.
	MaxBodySize int
}

// New creates a new extractor for the given domain.
func New(baseDomain string) *Extractor {
	return &Extractor{
		BaseDomain:  baseDomain,
		MaxBodySize: 1024 * 1024, // 1 MiB default
	}
}

// Extract parses HTML content and extracts page data.
func (e *Extractor) Extract(rawURL string, html []byte) (*Page, error) {
	// Limit HTML size
	if len(html) > e.MaxBodySize {
		html = html[:e.MaxBodySize]
	}

	doc, err := goquery.NewDocumentFromReader(strings.NewReader(string(html)))
	if err != nil {
		return nil, err
	}

	page := &Page{
		URL:   rawURL,
		Links: make([]string, 0),
	}

	// Extract title
	page.Title = e.extractTitle(doc)

	// Extract body text
	page.Body = e.extractBody(doc)

	// Extract links
	page.Links = e.extractLinks(doc, rawURL)

	return page, nil
}

// extractTitle extracts the page title from the document.
func (e *Extractor) extractTitle(doc *goquery.Document) string {
	title := doc.Find("title").First().Text()
	title = strings.TrimSpace(title)
	title = normalizeWhitespace(title)
	return title
}

// extractBody extracts readable text content from the document.
// Removes scripts, styles, and other non-content elements.
func (e *Extractor) extractBody(doc *goquery.Document) string {
	// Clone the document to avoid modifying the original
	clone := doc.Clone()

	// Remove non-content elements
	clone.Find("script").Remove()
	clone.Find("style").Remove()
	clone.Find("noscript").Remove()
	clone.Find("nav").Remove()
	clone.Find("footer").Remove()
	clone.Find("header").Remove()
	clone.Find("aside").Remove()
	clone.Find("[role='navigation']").Remove()
	clone.Find("[role='banner']").Remove()
	clone.Find("[role='contentinfo']").Remove()

	// Try to find main content area
	var bodyText string

	// Priority: main > article > #content > #main > body
	if main := clone.Find("main").First(); main.Length() > 0 {
		bodyText = main.Text()
	} else if article := clone.Find("article").First(); article.Length() > 0 {
		bodyText = article.Text()
	} else if content := clone.Find("#content, #main, .content, .main").First(); content.Length() > 0 {
		bodyText = content.Text()
	} else {
		bodyText = clone.Find("body").Text()
	}

	// Clean up the text
	bodyText = normalizeWhitespace(bodyText)

	// Limit size
	if len(bodyText) > e.MaxBodySize {
		bodyText = bodyText[:e.MaxBodySize]
	}

	return bodyText
}

// extractLinks extracts all same-domain links from the document.
func (e *Extractor) extractLinks(doc *goquery.Document, rawURL string) []string {
	links := make([]string, 0)
	seen := make(map[string]struct{})

	baseURL, err := url.Parse(rawURL)
	if err != nil {
		return links
	}

	doc.Find("a[href]").Each(func(i int, s *goquery.Selection) {
		href, exists := s.Attr("href")
		if !exists {
			return
		}

		// Resolve relative URLs
		parsed, err := url.Parse(href)
		if err != nil {
			return
		}

		resolved := baseURL.ResolveReference(parsed)

		// Only include http/https links
		if resolved.Scheme != "http" && resolved.Scheme != "https" {
			return
		}

		// Only include links to the same domain
		if resolved.Host != e.BaseDomain {
			return
		}

		// Remove fragment
		resolved.Fragment = ""

		// Normalize
		normalized := resolved.String()
		if _, ok := seen[normalized]; !ok {
			seen[normalized] = struct{}{}
			links = append(links, normalized)
		}
	})

	return links
}

// normalizeWhitespace collapses multiple whitespace characters into single spaces.
func normalizeWhitespace(s string) string {
	// Replace all whitespace sequences with a single space
	re := regexp.MustCompile(`\s+`)
	s = re.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

// IsHTML checks if content type indicates HTML content.
func IsHTML(contentType string) bool {
	contentType = strings.ToLower(contentType)
	return strings.Contains(contentType, "text/html") ||
		strings.Contains(contentType, "application/xhtml+xml")
}
