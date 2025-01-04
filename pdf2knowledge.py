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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract knowledge and summarize the following text:"},
                {"role": "user", "content": chunk}
            ]
        )
        knowledge_base.append(response.choices[0].message.content)
    return knowledge_base

# Main function
def main():
    pdf_folder = os.path.join(os.getcwd(), "pdf")
    output_folder = os.path.join(os.getcwd(), "extracted-knowledge")
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    pdf_count = len(pdf_files)

    if not pdf_files:
        print("No PDF files found in the ./pdf folder.")
        return
    
    batch_size = 5

    for i in range(0, pdf_count, batch_size):
        batch = pdf_files[i:i+batch_size]
        output_file = os.path.join(output_folder, f"extracted_knowledge_{i//batch_size + 1}.txt")

        with open(output_file, "w", encoding="utf-8") as out_file:
            for file_name in batch:
                pdf_path = os.path.join(pdf_folder, file_name)

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

        print(f"Batch {i//5 + 1} knowledge saved to {output_file}")

if __name__ == "__main__":
    main()
