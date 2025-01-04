import os
import PyPDF2
from openai import OpenAI

client = OpenAI()

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to preprocess the text
def preprocess_text(text):
    # Clean and structure the text
    text = text.replace("\n", " ").strip()  # Remove newlines
    return text

# Function to split text into chunks suitable for a model
def chunk_text(text, max_chunk_size=2000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        if current_size + len(word) + 1 <= max_chunk_size:
            current_chunk.append(word)
            current_size += len(word) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to generate embeddings or interact with the GPT model
def process_with_gpt(chunks):
    knowledge_base = []
    for chunk in chunks:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract knowledge and summarize the following text:"},
            {"role": "user", "content": chunk}
        ])
        knowledge_base.append(response.choices[0].message.content)
    return knowledge_base

# Main function
def main():
    folder_path = os.getcwd()  # Current folder
    output_file = os.path.join(folder_path, "extracted_knowledge.txt")

    with open(output_file, "w", encoding="utf-8") as out_file:
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, file_name)

                print(f"Processing: {file_name}")

                # Step 1: Extract text
                pdf_text = extract_text_from_pdf(pdf_path)

                # Step 2: Preprocess the text
                clean_text = preprocess_text(pdf_text)

                # Step 3: Split into chunks
                text_chunks = chunk_text(clean_text)

                # Step 4: Process with GPT to build knowledge
                knowledge_base = process_with_gpt(text_chunks)

                # Step 5: Write the knowledge to the output file
                out_file.write(f"Knowledge from {file_name}:\n")
                for knowledge in knowledge_base:
                    out_file.write(knowledge + "\n\n")
                out_file.write("\n" + "-"*50 + "\n\n")

    print(f"Knowledge has been extracted and saved to {output_file}")

# Run the script
if __name__ == "__main__":
    main()

