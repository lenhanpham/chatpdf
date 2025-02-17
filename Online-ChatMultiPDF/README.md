# PDF Chat Using Model API

## Overview
The PDF Chat Model API is a web application that allows users to upload PDF documents and ask questions about their content. The application utilizes various AI models to generate answers based on the information extracted from the uploaded PDFs. Your PDF files are stored offline and totally safe.

![image-20250210234549801](UI.png)

## Project Structure
```
src/
├── services/
│   ├── answer_generation.py  # Handles answer generation logic
│   ├── model_handler.py      # Handles different model API interactions
│   └── vector_store.py       # Handles vector store operations
├── config/
│   └── settings.py           # Centralized configuration
├── utils/
│   └── logging_config.py     # Logging configuration
└── chatpdf.py               # Main application file
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd Online-ChatMultiPDF
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration
- Create a `.env` file in the root directory and add your API keys and other configuration settings as needed. Refer to the `settings.py` file for the required variables.

## Usage
1. Run the application:
   ```
   streamlit run src/chatpdf.py
   ```
2. Open your web browser and go to `http://localhost:8501` to access the application.
3. Upload a PDF document and start asking questions about its content.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.