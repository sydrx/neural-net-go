package nn

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// MNIST holds training and test splits.
type MNIST struct {
	TrainX, TrainY *Matrix // [60000 × 784] and [60000 × 10]
	TestX, TestY   *Matrix // [10000 × 784] and [10000 × 10]
}

// mnistFile describes one remote MNIST file.
type mnistFile struct {
	name   string
	url    string
	isImg  bool
	count  int
}

var mnistFiles = []mnistFile{
	{"train-images-idx3-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", true, 60000},
	{"train-labels-idx1-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", false, 60000},
	{"t10k-images-idx3-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", true, 10000},
	{"t10k-labels-idx1-ubyte.gz", "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz", false, 10000},
}

// LoadMNIST downloads (if needed) and parses the MNIST dataset into Matrices.
//   dir – local directory to cache the raw files (e.g. "data/mnist")
//
// Images are normalised to [0, 1]. Labels are one-hot encoded.
func LoadMNIST(dir string) (*MNIST, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	data := make([][]byte, len(mnistFiles))
	for i, f := range mnistFiles {
		path := filepath.Join(dir, f.name)
		raw, err := readOrDownload(path, f.url)
		if err != nil {
			return nil, fmt.Errorf("mnist: %w", err)
		}
		data[i] = raw
	}

	trainImgs, err := parseImages(data[0])
	if err != nil { return nil, err }
	trainLbls, err := parseLabels(data[1])
	if err != nil { return nil, err }
	testImgs, err := parseImages(data[2])
	if err != nil { return nil, err }
	testLbls, err := parseLabels(data[3])
	if err != nil { return nil, err }

	return &MNIST{
		TrainX: toImageMatrix(trainImgs),
		TrainY: toLabelMatrix(trainLbls),
		TestX:  toImageMatrix(testImgs),
		TestY:  toLabelMatrix(testLbls),
	}, nil
}

// ── Parsing ───────────────────────────────────────────────────────────────────

func parseImages(gz []byte) ([][]byte, error) {
	r, err := gzipReader(gz)
	if err != nil { return nil, err }
	defer r.Close()

	var magic, n, rows, cols int32
	binary.Read(r, binary.BigEndian, &magic)
	binary.Read(r, binary.BigEndian, &n)
	binary.Read(r, binary.BigEndian, &rows)
	binary.Read(r, binary.BigEndian, &cols)
	if magic != 2051 {
		return nil, fmt.Errorf("bad image magic %d", magic)
	}

	size := int(rows * cols)
	imgs := make([][]byte, n)
	for i := range imgs {
		imgs[i] = make([]byte, size)
		if _, err := io.ReadFull(r, imgs[i]); err != nil {
			return nil, err
		}
	}
	return imgs, nil
}

func parseLabels(gz []byte) ([]byte, error) {
	r, err := gzipReader(gz)
	if err != nil { return nil, err }
	defer r.Close()

	var magic, n int32
	binary.Read(r, binary.BigEndian, &magic)
	binary.Read(r, binary.BigEndian, &n)
	if magic != 2049 {
		return nil, fmt.Errorf("bad label magic %d", magic)
	}

	labels := make([]byte, n)
	_, err = io.ReadFull(r, labels)
	return labels, err
}

func toImageMatrix(imgs [][]byte) *Matrix {
	N, D := len(imgs), len(imgs[0])
	m := NewMatrix(N, D)
	for i, img := range imgs {
		for j, px := range img {
			m.Set(i, j, float64(px)/255.0)
		}
	}
	return m
}

func toLabelMatrix(labels []byte) *Matrix {
	m := NewMatrix(len(labels), 10)
	for i, l := range labels {
		m.Set(i, int(l), 1.0) // one-hot
	}
	return m
}

// ── I/O helpers ───────────────────────────────────────────────────────────────

func readOrDownload(path, url string) ([]byte, error) {
	if b, err := os.ReadFile(path); err == nil {
		fmt.Printf("  ✓ cached  %s\n", filepath.Base(path))
		return b, nil
	}
	fmt.Printf("  ↓ downloading %s ...\n", filepath.Base(path))
	resp, err := http.Get(url)
	if err != nil { return nil, err }
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil { return nil, err }
	return b, os.WriteFile(path, b, 0644)
}

func gzipReader(data []byte) (io.ReadCloser, error) {
	// io.NopCloser trick: wrap bytes as io.Reader first
	r, err := gzip.NewReader(readerFromBytes(data))
	return r, err
}

type byteReader struct{ data []byte; pos int }
func readerFromBytes(b []byte) io.Reader { return &byteReader{data: b} }
func (r *byteReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) { return 0, io.EOF }
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}
