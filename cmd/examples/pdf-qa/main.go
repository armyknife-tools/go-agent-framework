// cmd/examples/pdf-qa/main.go

package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"hash/fnv"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode"

	"github.com/armyknife-tools/go-agent-framework/pkg/chains/qa"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/document"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/prompt"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/retriever"
	"github.com/armyknife-tools/go-agent-framework/pkg/core/vectorstore"
	"github.com/armyknife-tools/go-agent-framework/pkg/llms/anthropic"

	"github.com/ledongthuc/pdf"
)

var (
	apiKey       = flag.String("api-key", "", "Anthropic API key")
	query        = flag.String("query", "", "Query to answer")
	pdfDir       = flag.String("pdf-dir", "", "Directory containing PDF files")
	maxPages     = flag.Int("max-pages", 50, "Maximum number of pages to process per PDF")
	numResults   = flag.Int("results", 5, "Number of document chunks to retrieve")
	chunkSize    = flag.Int("chunk-size", 1000, "Size of text chunks to split documents into")
	chunkOverlap = flag.Int("chunk-overlap", 200, "Overlap between text chunks")
	debug        = flag.Bool("debug", false, "Enable debug output")
)

// Special tokens for embedding
const (
	titlePrefix    = "TITLE: "
	contentPrefix  = "CONTENT: "
	metadataPrefix = "METADATA: "
)

func main() {
	flag.Parse()

	// Check if API key is provided
	if *apiKey == "" {
		// Try to get API key from environment
		*apiKey = os.Getenv("ANTHROPIC_API_KEY")
		if *apiKey == "" {
			log.Fatal("API key not provided. Use --api-key flag or ANTHROPIC_API_KEY environment variable.")
		}
	}

	// Check if PDF directory is provided
	if *pdfDir == "" {
		log.Fatal("PDF directory not provided. Use --pdf-dir flag.")
	}

	// Create a context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	// Create the LLM
	llm := anthropic.NewAnthropicLLM(*apiKey,
		anthropic.WithModel("claude-3-7-sonnet-20250219"),
		anthropic.WithTemperature(0.0),
		anthropic.WithMaxTokens(2000),
	)

	fmt.Println("PDF Question Answering System")
	fmt.Println("============================")

	// Load PDF files from the directory
	fmt.Printf("Loading PDFs from %s...\n", *pdfDir)
	docs, err := readPDFsFromDirectory(*pdfDir, *maxPages)
	if err != nil {
		log.Fatalf("Error loading PDFs: %v", err)
	}

	if len(docs) == 0 {
		log.Fatalf("No PDFs found in directory: %s", *pdfDir)
	}

	fmt.Printf("Loaded %d document pages from PDFs\n", len(docs))

	// Create text chunks from the documents
	chunks := splitDocumentsIntoChunks(docs, *chunkSize, *chunkOverlap)
	fmt.Printf("Created %d text chunks for embedding\n", len(chunks))

	// Create an enhanced embedding model
	embedder := &EnhancedEmbedder{}

	// Create a vector store and add the document chunks
	store := vectorstore.NewMemoryVectorStore(embedder)

	// Create a document filter to remove empty or very short chunks
	validChunks := []*document.Document{}
	for _, chunk := range chunks {
		if len(strings.TrimSpace(chunk.PageContent)) > 50 {
			validChunks = append(validChunks, chunk)
		}
	}

	// Add documents to the vector store
	fmt.Println("Adding documents to vector store...")
	err = store.AddDocuments(ctx, validChunks)
	if err != nil {
		log.Fatalf("Error adding documents to vector store: %v", err)
	}

	// Create a custom filter function for metadata-based filtering
	metadataFilter := func(metadata map[string]interface{}) bool {
		// This is a placeholder for custom filtering logic
		// For example, you could filter by document type, date, etc.
		return true
	}

	// Create a retriever with enhanced settings
	vectorRetriever := retriever.NewVectorStoreRetriever(
		store,
		retriever.WithK(*numResults),
		retriever.WithFilter(metadataFilter),
	)

	// Create a QA prompt template with instructions for using citations
	customPrompt := `You are a helpful assistant that answers questions based on the provided documents.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite your sources by indicating which document and page number the information came from.

{context}

Question: {question}
Helpful Answer:`

	qaPrompt, err := prompt.NewPromptTemplate(customPrompt, []string{"context", "question"})
	if err != nil {
		log.Fatalf("Error creating QA prompt: %v", err)
	}

	// Create a QA chain
	qaChain, err := qa.NewQAChain(qa.QAChainOptions{
		Retriever:  vectorRetriever,
		LLM:        llm,
		Prompt:     qaPrompt,
		ReturnDocs: true,
	})
	if err != nil {
		log.Fatalf("Error creating QA chain: %v", err)
	}

	// Interactive question loop
	reader := bufio.NewReader(os.Stdin)
	for {
		// If query is provided as a flag, use it; otherwise, prompt the user
		var userQuestion string
		if *query != "" {
			userQuestion = *query
			*query = "" // Clear the flag after using it once
		} else {
			fmt.Print("\nEnter your question (or type 'exit' to quit): ")
			input, err := reader.ReadString('\n')
			if err != nil {
				log.Fatalf("Error reading input: %v", err)
			}
			userQuestion = strings.TrimSpace(input)
			if userQuestion == "exit" {
				break
			}
		}

		// Skip empty questions
		if userQuestion == "" {
			continue
		}

		fmt.Printf("\nQuestion: %s\n\n", userQuestion)

		// Check if we have documents before running the query
		if len(validChunks) == 0 {
			fmt.Println("WARNING: No documents in the vector store. The LLM will not have any context to work with.")
		}

		// Time the query execution
		startTime := time.Now()

		// Run the QA chain
		result, err := qaChain.Run(ctx, qa.QAInput{
			Question: userQuestion,
		})
		if err != nil {
			fmt.Printf("Error running QA chain: %v\n", err)
			continue
		}

		// Print the answer
		fmt.Println("Answer:", result.Answer)
		fmt.Printf("\nTime taken: %.2f seconds\n\n", time.Since(startTime).Seconds())

		// Print source documents if available
		if result.SourceDocs != nil && len(result.SourceDocs) > 0 {
			fmt.Println("Sources:")
			for i, doc := range result.SourceDocs {
				fmt.Printf("Document %d:", i+1)

				// Print source information
				source, hasSource := doc.Metadata["source"]

				if hasSource {
					fmt.Printf(" (Source: %v", source)

					if page, hasPage := doc.Metadata["page"]; hasPage {
						fmt.Printf(", Page: %v", page)
					}

					if title, hasTitle := doc.Metadata["title"]; hasTitle && title != "" {
						fmt.Printf(", Title: %v", title)
					}

					fmt.Printf(")")
				}
				fmt.Println()
				
				// Print a snippet of the document
				const maxLength = 300
				content := doc.PageContent
				if len(content) > maxLength {
					content = content[:maxLength] + "..."
				}
				fmt.Printf("  %s\n\n", content)
			}
		} else {
			fmt.Println("No source documents were retrieved for this query.")
		}

		// Debug information
		if *debug {
			fmt.Println("\nDebug Information:")
			fmt.Printf("Total documents available: %d\n", len(chunks))
			fmt.Printf("Documents retrieved: %d\n", len(result.SourceDocs))
			fmt.Printf("Vector store document count: %d\n", len(validChunks))
			fmt.Println("Retrieval parameters:")
			fmt.Printf("  Results: %d\n", *numResults)
			fmt.Printf("  Chunk size: %d\n", *chunkSize)
			fmt.Printf("  Chunk overlap: %d\n", *chunkOverlap)
		}
	}
}

// readPDFsFromDirectory reads all PDF files from a directory and converts them to Document objects.
func readPDFsFromDirectory(dirPath string, maxPages int) ([]*document.Document, error) {
	files, err := ioutil.ReadDir(dirPath)
	if err != nil {
		return nil, fmt.Errorf("error reading directory: %w", err)
	}

	var allDocs []*document.Document
	var pdfFiles []os.FileInfo

	// Filter for PDF files
	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(strings.ToLower(file.Name()), ".pdf") {
			pdfFiles = append(pdfFiles, file)
		}
	}

	if len(pdfFiles) == 0 {
		return nil, fmt.Errorf("no PDF files found in directory")
	}

	// Process each PDF file
	for _, file := range pdfFiles {
		filePath := filepath.Join(dirPath, file.Name())
		fmt.Printf("Processing %s...\n", file.Name())

		// Extract title from filename by removing extension and converting to title case
		title := strings.TrimSuffix(file.Name(), filepath.Ext(file.Name()))
		title = toTitleCase(title)

		// Read the PDF with enhanced extraction
		pdfDocs, err := readPDFEnhanced(filePath, maxPages, title)
		if err != nil {
			fmt.Printf("Error reading %s: %v (skipping)\n", file.Name(), err)
			continue
		}

		allDocs = append(allDocs, pdfDocs...)
	}

	return allDocs, nil
}

// readPDFEnhanced reads a PDF file with enhanced text extraction and converts it to Document objects.
// It attempts multiple extraction methods to get the best results.
func readPDFEnhanced(filePath string, maxPages int, title string) ([]*document.Document, error) {
	f, r, err := pdf.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening PDF: %w", err)
	}
	defer f.Close()

	totalPages := r.NumPage()
	if maxPages > 0 && totalPages > maxPages {
		totalPages = maxPages
	}

	fmt.Printf("PDF has %d pages, processing up to %d pages\n", r.NumPage(), totalPages)

	var docs []*document.Document
	var successfulPages int

	// Process each page
	for pageNum := 1; pageNum <= totalPages; pageNum++ {
		fmt.Printf("Processing page %d...\n", pageNum)

		p := r.Page(pageNum)
		if p.V.IsNull() {
			fmt.Printf("Page %d is null, skipping\n", pageNum)
			continue
		}

		// Try multiple extraction methods
		var pageText string
		var extractionSuccess bool

		// Method 1: GetPlainText
		text, err := p.GetPlainText(nil)
		if err == nil && len(strings.TrimSpace(text)) > 50 {
			pageText = text
			extractionSuccess = true
			fmt.Printf("Page %d: Extracted %d characters of text using plain text method\n", pageNum, len(pageText))
		} else {
			fmt.Printf("Plain text extraction for page %d failed or returned minimal text, trying alternative methods\n", pageNum)
			
			// Method 2: GetTextByColumn
			columns, err := p.GetTextByColumn()
			if err == nil && len(columns) > 0 {
				var columnTexts []string
				for _, col := range columns {
					var texts []string
					for _, text := range col.Content {
						texts = append(texts, text.S)
					}
					columnTexts = append(columnTexts, strings.Join(texts, " "))
				}
				pageText = strings.Join(columnTexts, " ")
				if len(strings.TrimSpace(pageText)) > 50 {
					extractionSuccess = true
					fmt.Printf("Page %d: Extracted %d characters of text using column method\n", pageNum, len(pageText))
				}
			}
			
			// Method 3: GetTextByRow
			if !extractionSuccess {
				rows, err := p.GetTextByRow()
				if err == nil && len(rows) > 0 {
					var rowTexts []string
					for _, row := range rows {
						var texts []string
						for _, text := range row.Content {
							texts = append(texts, text.S)
						}
						rowTexts = append(rowTexts, strings.Join(texts, " "))
					}
					pageText = strings.Join(rowTexts, " ")
					if len(strings.TrimSpace(pageText)) > 50 {
						extractionSuccess = true
						fmt.Printf("Page %d: Extracted %d characters of text using row method\n", pageNum, len(pageText))
					}
				}
			}
		}

		if !extractionSuccess {
			fmt.Printf("All text extraction methods failed for page %d\n", pageNum)
			continue
		}

		// Clean up text
		pageText = cleanText(pageText)
		if pageText == "" {
			fmt.Printf("Page %d has no text after cleaning, skipping\n", pageNum)
			continue
		}

		// Create a document for this page
		doc := document.NewDocument(pageText, map[string]interface{}{
			"source": filepath.Base(filePath),
			"page":   pageNum,
			"path":   filePath,
			"title":  title,
		})

		docs = append(docs, doc)
		successfulPages++
	}

	if len(docs) == 0 {
		return nil, fmt.Errorf("no text extracted from PDF")
	}

	fmt.Printf("Successfully extracted text from %d pages\n", successfulPages)
	return docs, nil
}

// toTitleCase converts a string to title case.
func toTitleCase(s string) string {
	// Replace underscores and hyphens with spaces
	s = strings.ReplaceAll(s, "_", " ")
	s = strings.ReplaceAll(s, "-", " ")

	words := strings.Fields(s)
	for i, word := range words {
		if len(word) == 0 {
			continue
		}

		// Convert the first letter to uppercase and the rest to lowercase
		runes := []rune(word)
		runes[0] = unicode.ToUpper(runes[0])
		for j := 1; j < len(runes); j++ {
			runes[j] = unicode.ToLower(runes[j])
		}
		words[i] = string(runes)
	}

	return strings.Join(words, " ")
}

// readPDF reads a PDF file and converts it to Document objects.
func readPDF(filePath string, maxPages int, title string) ([]*document.Document, error) {
	f, r, err := pdf.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening PDF: %w", err)
	}
	defer f.Close()

	var docs []*document.Document

	totalPages := r.NumPage()
	if maxPages > 0 && totalPages > maxPages {
		totalPages = maxPages
	}

	fmt.Printf("PDF has %d pages, processing up to %d pages\n", r.NumPage(), totalPages)

	// Extract text from each page
	for pageNum := 1; pageNum <= totalPages; pageNum++ {
		fmt.Printf("Processing page %d...\n", pageNum)
		
		p := r.Page(pageNum)
		if p.V.IsNull() {
			fmt.Printf("Page %d is null, skipping\n", pageNum)
			continue
		}

		// Try to extract text using the plain text method
		text, err := p.GetPlainText(nil)
		if err != nil || len(strings.TrimSpace(text)) < 50 {
			// If plain text extraction failed or returned very little text, try alternative methods
			fmt.Printf("Plain text extraction for page %d failed or returned minimal text, trying alternative methods\n", pageNum)
			
			// Try getting text by column
			columns, err := p.GetTextByColumn()
			if err == nil && len(columns) > 0 {
				var textBuilder strings.Builder
				for _, column := range columns {
					for _, text := range column.Content {
						textBuilder.WriteString(text.S + " ")
					}
				}
				text = textBuilder.String()
				fmt.Printf("Successfully extracted text by column from page %d\n", pageNum)
			} else {
				// Try getting text by row
				rows, err := p.GetTextByRow()
				if err == nil && len(rows) > 0 {
					var textBuilder strings.Builder
					for _, row := range rows {
						for _, text := range row.Content {
							textBuilder.WriteString(text.S + " ")
						}
						textBuilder.WriteString("\n")
					}
					text = textBuilder.String()
					fmt.Printf("Successfully extracted text by row from page %d\n", pageNum)
				} else {
					fmt.Printf("All text extraction methods failed for page %d\n", pageNum)
				}
			}
		}

		// Clean up text
		text = cleanText(text)
		if text == "" {
			fmt.Printf("Page %d has no text after cleaning, skipping\n", pageNum)
			continue
		}

		fmt.Printf("Page %d: Extracted %d characters of text\n", pageNum, len(text))
		
		// Create a document for this page
		doc := document.NewDocument(text, map[string]interface{}{
			"source": filepath.Base(filePath),
			"page":   pageNum,
			"path":   filePath,
			"title":  title,
		})

		docs = append(docs, doc)
	}

	if len(docs) == 0 {
		fmt.Println("WARNING: No text was extracted from the PDF!")
	} else {
		fmt.Printf("Successfully extracted text from %d pages\n", len(docs))
	}

	return docs, nil
}

// cleanText performs text cleanup operations.
func cleanText(text string) string {
	// Remove excessive whitespace
	re := regexp.MustCompile(`\s+`)
	text = re.ReplaceAllString(text, " ")

	// Remove non-printable characters
	text = strings.Map(func(r rune) rune {
		if unicode.IsPrint(r) || unicode.IsSpace(r) {
			return r
		}
		return -1
	}, text)

	return strings.TrimSpace(text)
}

// splitDocumentsIntoChunks splits documents into smaller chunks with semantic boundaries.
func splitDocumentsIntoChunks(docs []*document.Document, chunkSize, overlap int) []*document.Document {
	var chunks []*document.Document

	for _, doc := range docs {
		text := doc.PageContent

		// Skip if text is shorter than chunk size
		if len(text) < chunkSize {
			chunks = append(chunks, doc)
			continue
		}

		// Try to split at paragraph boundaries first
		paragraphs := splitIntoParagraphs(text)

		// If we have very few paragraphs, split at sentence boundaries
		if len(paragraphs) <= 2 {
			paragraphs = splitIntoSentences(text)
		}

		// Build chunks from paragraphs
		var currentChunk strings.Builder
		var currentMetadata map[string]interface{}

		for i, para := range paragraphs {
			// Initialize metadata from the first paragraph
			if i == 0 || currentMetadata == nil {
				currentMetadata = make(map[string]interface{})
				for k, v := range doc.Metadata {
					currentMetadata[k] = v
				}
				// Add chunk number to metadata
				currentMetadata["chunk"] = len(chunks) + 1
			}

			// If adding this paragraph would exceed chunk size, create a new chunk
			if currentChunk.Len()+len(para) > chunkSize && currentChunk.Len() > 0 {
				// Create a chunk from accumulated text
				chunk := document.NewDocument(currentChunk.String(), currentMetadata)
				chunks = append(chunks, chunk)

				// Start a new chunk with overlap
				overlapStart := max(0, currentChunk.Len()-overlap)
				currentChunk = strings.Builder{}
				currentChunk.WriteString(currentChunk.String()[overlapStart:])

				// Create new metadata for the next chunk
				currentMetadata = make(map[string]interface{})
				for k, v := range doc.Metadata {
					currentMetadata[k] = v
				}
				currentMetadata["chunk"] = len(chunks) + 1
			}

			// Add paragraph to current chunk
			currentChunk.WriteString(para)
			currentChunk.WriteString(" ")
		}

		// Add the last chunk if it has content
		if currentChunk.Len() > 0 {
			chunk := document.NewDocument(currentChunk.String(), currentMetadata)
			chunks = append(chunks, chunk)
		}
	}

	return chunks
}

// splitIntoParagraphs splits text into paragraphs.
func splitIntoParagraphs(text string) []string {
	// Split by double newlines or multiple spaces after period
	paragraphSplitter := regexp.MustCompile(`\n\s*\n|\.\s{2,}`)
	paragraphs := paragraphSplitter.Split(text, -1)

	// Filter out empty paragraphs
	var nonEmptyParagraphs []string
	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p != "" {
			nonEmptyParagraphs = append(nonEmptyParagraphs, p)
		}
	}

	return nonEmptyParagraphs
}

// splitIntoSentences splits text into sentences.
func splitIntoSentences(text string) []string {
	// This is a simplified sentence splitter
	// In a production system, you might want a more sophisticated NLP-based splitter
	sentenceSplitter := regexp.MustCompile(`[.!?]\s+`)
	sentences := sentenceSplitter.Split(text, -1)

	// Filter out empty sentences
	var nonEmptySentences []string
	for _, s := range sentences {
		s = strings.TrimSpace(s)
		if s != "" {
			nonEmptySentences = append(nonEmptySentences, s)
		}
	}

	return nonEmptySentences
}

// max returns the maximum of two integers.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// EnhancedEmbedder is an improved embedding implementation.
type EnhancedEmbedder struct{}

// EmbedDocuments embeds multiple documents.
func (e *EnhancedEmbedder) EmbedDocuments(ctx context.Context, documents []string) ([][]float32, error) {
	embeddings := make([][]float32, len(documents))
	for i, doc := range documents {
		embeddings[i] = e.createEnhancedEmbedding(doc)
	}
	return embeddings, nil
}

// EmbedQuery embeds a query.
func (e *EnhancedEmbedder) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	// Process query to enhance semantic matching
	expandedQuery := e.expandQuery(query)
	return e.createEnhancedEmbedding(expandedQuery), nil
}

// expandQuery adds relevant terms to improve retrieval.
func (e *EnhancedEmbedder) expandQuery(query string) string {
	// This is a simple implementation - in production, you'd use
	// a more sophisticated query expansion method

	// Add common synonyms for certain terms
	query = strings.ToLower(query)

	// Convert to lowercase and trim spaces
	query = strings.TrimSpace(strings.ToLower(query))

	// Extract key terms - a simple approach using word tokenization
	words := strings.Fields(query)
	var keyTerms []string

	// Filter out stopwords and keep meaningful terms
	stopwords := map[string]bool{
		"a": true, "an": true, "the": true, "and": true, "or": true,
		"for": true, "in": true, "on": true, "at": true, "to": true,
		"is": true, "are": true, "was": true, "were": true,
		"what": true, "when": true, "where": true, "why": true, "how": true,
	}

	for _, word := range words {
		if !stopwords[word] && len(word) > 2 {
			keyTerms = append(keyTerms, word)
		}
	}

	// Make sure we're also including the exact key terms for precise matching
	if len(keyTerms) > 0 {
		return fmt.Sprintf("%s %s", query, strings.Join(keyTerms, " "))
	}

	return query
}

// createEnhancedEmbedding creates a more sophisticated embedding.
func (e *EnhancedEmbedder) createEnhancedEmbedding(text string) []float32 {
	// This is still a simplified embedding method but with improvements
	// In production, you'd want to use an actual embedding model API

	// Use a larger dimension for embeddings
	dim := 256
	embedding := make([]float32, dim)

	// If text is empty, return zero embedding
	if text == "" {
		return embedding
	}

	// Normalize and clean text
	text = strings.ToLower(text)

	// Extract different features from the text

	// 1. Character n-grams (first 64 dimensions)
	for i := 1; i <= 3; i++ { // 1-gram, 2-gram, 3-gram
		ngrams := generateNGrams(text, i)
		for j := 0; j < min(64, len(ngrams)); j++ {
			// Hash the n-gram to a stable index
			idx := int(hashString(ngrams[j])) % 64
			embedding[idx] += 1.0
		}
	}

	// 2. Word features (next 64 dimensions)
	words := strings.Fields(text)

	// Word count (normalized)
	if len(words) > 0 {
		embedding[64] = float32(min(len(words), 1000)) / 1000.0
	}

	// Word-level n-grams
	wordNgrams := generateWordNGrams(words, 2) // bigrams
	for i := 0; i < min(63, len(wordNgrams)); i++ {
		idx := 65 + (int(hashString(wordNgrams[i])) % 63)
		embedding[idx] += 1.0
	}

	// 3. Semantic features (next 64 dimensions)
	// Count occurrences of semantic categories
	categoryWords := map[string]int{
		// These are just examples - expand for your domain
		"technology": 128,
		"science":    129,
		"business":   130,
		"finance":    131,
		"health":     132,
		"medical":    133,
		"history":    134,
		"politics":   135,
		"art":        136,
		"music":      137,
	}

	for word, idx := range categoryWords {
		if strings.Contains(text, word) {
			embedding[idx] += 1.0
		}
	}

	// 4. Statistical features (next 64 dimensions)
	// Document length (normalized)
	embedding[192] = float32(min(len(text), 10000)) / 10000.0

	// Average word length
	totalWordLength := 0
	for _, word := range words {
		totalWordLength += len(word)
	}
	if len(words) > 0 {
		avgWordLength := float32(totalWordLength) / float32(len(words))
		embedding[193] = avgWordLength / 20.0 // Normalize
	}

	// Sentence count (normalized)
	sentences := strings.Count(text, ". ") + strings.Count(text, "! ") + strings.Count(text, "? ")
	embedding[194] = float32(min(sentences, 100)) / 100.0

	// Normalize the embedding to unit length
	var sum float32
	for _, val := range embedding {
		sum += val * val
	}

	if sum > 0 {
		magnitude := float32(math.Sqrt(float64(sum)))
		for i := range embedding {
			embedding[i] /= magnitude
		}
	}

	return embedding
}

// generateNGrams creates character n-grams from text.
func generateNGrams(text string, n int) []string {
	var ngrams []string
	runes := []rune(text)

	for i := 0; i <= len(runes)-n; i++ {
		ngrams = append(ngrams, string(runes[i:i+n]))
	}

	return ngrams
}

// generateWordNGrams creates word n-grams from a slice of words.
func generateWordNGrams(words []string, n int) []string {
	var ngrams []string

	for i := 0; i <= len(words)-n; i++ {
		ngram := strings.Join(words[i:i+n], " ")
		ngrams = append(ngrams, ngram)
	}

	return ngrams
}

// hashString creates a hash of a string.
func hashString(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// BatchSize returns the maximum batch size for embedding operations.
func (e *EnhancedEmbedder) BatchSize() int {
	return 100
}

// ModelName returns the name of the embedding model.
func (e *EnhancedEmbedder) ModelName() string {
	return "enhanced-embedder"
}

// Dimension returns the dimension of the embedding vectors.
func (e *EnhancedEmbedder) Dimension() int {
	return 256
}
