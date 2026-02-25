## Quick Start (Local Renderer Path -- Low VRAM)

### Data Preparation Instructions:
1. **Download Example Data:**  
   To get started, download the Shakespeare dataset using the following command:
   ```bash
   curl -O http://example.com/path/to/shakespeare-dataset.zip
   ```
   This will download a ZIP file containing the necessary text files.

2. **Extract the Data:**
   Unzip the downloaded file:
   ```bash
   unzip shakespeare-dataset.zip
   cd shakespeare-dataset
   ```

3. **Prepare the HRM Index:**  
   Initialize the HRM index using the text data:
   ```bash
   hrm index --data-path ./shakespeare.txt --output-path ./shakespeare_index.hrm
   ```

### Usage Examples:
- To render the content using the HRM index, use:
  ```bash
  hrm render --index ./shakespeare_index.hrm --output ./render_output
  ```

- For more specific commands and options, consult the HRM documentation or run:
  ```bash
  hrm --help
  ```

These instructions should help you get started with rendering using a low VRAM path. Make sure the examples align with your setup and adjust paths as necessary.