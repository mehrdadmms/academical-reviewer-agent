
import openai
import os
from glob import glob
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI

# Function to load knowledge into a retriever
def create_retriever_from_file(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()
    return retriever

# Function to create a RetrievalQA chain
def create_retrieval_qa_chain(retriever, model_name="gpt-4o"):
    llm = ChatOpenAI(model_name=model_name)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# Function to iteratively fetch and compose the review
def iterative_review_generation(task, qa_chain, prv_result):
    print(f"Input size for current iteration: {len(task.split())}")
    response = qa_chain(task)
    review = response["result"]

    return review

# Main function
def main():
     # Path to the folder containing extracted knowledge files
    knowledge_folder = "extracted-knowledge"
    knowledge_file_paths = sorted(glob(os.path.join(knowledge_folder, "extracted_knowledge_*.txt")))

    if not knowledge_file_paths:
        print("No extracted knowledge files found!")
        return

    # Output file path
    output_file_path = "result.txt"
    word_goal=2000
    total_repeats = 1
    task = f"""

    Task:
    You're a prompt engineer working exclusively with voice AI. 
    - For each of the survey types create a AI Assistant prompt that prompts a voice AI Agent to act as an interviewer.
    In the end give me a list of all the survey types and their corresponding prompts.


    {{}}

    Survey types:
        1. **Questionnaires**  
            - Closed-forced choice
            - Open-broad
            - General-focused
            - Specific-focused
            - Factual
            - Hypothetical
            - Judgmental
            - Comparative
            - Neutral
            - Leading 
            - Blaring 
            - Request for suggestions
            - Request for questions  

        2. **Surveys**  
            - Questionnaires
            - Structured interviews  

        3. **Structured Interviews**
            - Conducted in person
            - By phone
            - Through various communication technologies

    Rules:
    1. Only use knowledge provided to write.
    2. Use example work as a reference and enrich it based on the knowledge provided and tailor for each survey type.
    3. Use the knowledge provided to write the prompts tailored to each survey type.
    4. Don't give me response as the interviewer agent, give me the prompt that must be fed to the AI agent as system prompt so later on it can interview the user.
    5. For each survey type, provide a set of rules specific to that survey type based on the knowledge provided that the interviewer agent must follow.
    6. Minimum set of rules per survey type is 5.
    7. Apart from survey type rules add 5 general rules that the interviewer agent must follow.
    8. The end result must have at least 10 rules (general and survey type specific) for each survey type.

    Example work: 
        You're a interviewer that ask the below questions to the interviewee, you should detect contrasts and ask for more explanation if needed. 
        say hello first before use start the conversation and start a warmup conversation then ask questions 
        double check the answer if the user said something that is not clear 
        Don't make assumptions, just ask the questions and try to get a clear answer. repeat the question if you didn't get a clear answer 
        Don't ask Irrelevant questions, just ask the questions that are related to the list of questions, if the user said something that is not related to the list of questions, guide the user to the list of questions 
        before start let them know that they're being recorded and the need to concent to be recorded 
        if you noticed sth strange in his answer, ask him to explain it more and confirm it with your understanding

    """

    words_written = 0
    result = ""
    for repeat in range(total_repeats):
        print(f"\n===== Starting Repeat {repeat + 1} of {total_repeats} =====\n")
        for file_path in knowledge_file_paths:
            print(f"Processing file: {file_path}")

            # Create a retriever for the current file
            retriever = create_retriever_from_file(file_path)

            # Create a RetrievalQA chain
            qa_chain = create_retrieval_qa_chain(retriever)
            # task = """
            #     give me numbered list of all the ways to write surveys to develop innovative ideas, research complex problems and design effective solutions.
            #     then for each go into details and give me list of types in bullet points.
            #     Only use knowledge provided to write.
            #     don't add any new information.
            #     don't repeat the same information.
            #     don't explain but tell me which survays are for goups and which ones are for individuals.
            # """
            # Generate the review for the current file
            result = iterative_review_generation(task, qa_chain, result)
            words_written = len(result.split())
            print(f"Iteration {repeat + 1} - file {file_path}: {words_written} words written.\n\n")
        # if words_written > word_goal and repeat >= 1:
            # break


    # Step 4: Save the result to a text file
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(result)

    print(f"Task completed. Result saved to {output_file_path}")

# Run the script
if __name__ == "__main__":
    main()