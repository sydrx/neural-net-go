package nn

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// Save writes all layer weights and biases to a binary file.
// Format: [num_layers] then for each layer: [rows][cols][...weights][num_biases][...biases]
func (n *Network) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// write number of layers
	if err := binary.Write(f, binary.LittleEndian, int64(len(n.layers))); err != nil {
		return err
	}

	for _, l := range n.layers {
		// write weight matrix shape
		binary.Write(f, binary.LittleEndian, int64(l.W.Rows))
		binary.Write(f, binary.LittleEndian, int64(l.W.Cols))
		// write weight data
		if err := binary.Write(f, binary.LittleEndian, l.W.Data); err != nil {
			return err
		}
		// write biases
		binary.Write(f, binary.LittleEndian, int64(len(l.B)))
		if err := binary.Write(f, binary.LittleEndian, l.B); err != nil {
			return err
		}
	}
	return nil
}

// LoadWeights reads weights from a file into an existing network.
// The network must already have the same architecture.
func (n *Network) LoadWeights(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var numLayers int64
	binary.Read(f, binary.LittleEndian, &numLayers)
	if int(numLayers) != len(n.layers) {
		return fmt.Errorf("load: expected %d layers, file has %d", len(n.layers), numLayers)
	}

	for _, l := range n.layers {
		var rows, cols int64
		binary.Read(f, binary.LittleEndian, &rows)
		binary.Read(f, binary.LittleEndian, &cols)

		if int(rows) != l.W.Rows || int(cols) != l.W.Cols {
			return fmt.Errorf("load: weight shape mismatch")
		}
		if err := binary.Read(f, binary.LittleEndian, l.W.Data); err != nil && err != io.EOF {
			return err
		}

		var nb int64
		binary.Read(f, binary.LittleEndian, &nb)
		if err := binary.Read(f, binary.LittleEndian, l.B); err != nil && err != io.EOF {
			return err
		}
	}
	return nil
}
