
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

    prev = "" if prv_result == "" else "The following was written in the previous iteration, use the knowledge to enrich it to reach to 2500 words: \n" + prv_result
    itr_task = task.format(prev)
    print(f"Input size for current iteration: {len(itr_task.split())}")
    response = qa_chain(itr_task)
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
    total_repeats = 3
    task = f"""

    Task:
    Write a 2500-word critical review exploring financing a new venture. 
    The review should focus on the strengths and weaknesses of the research theme, including venture capital, corporate venture capital, angel investment, crowdfunding, accelerators, Friends and Family, Bank Loans, grants, Strategic Partnerships, and blockchain technology in financing a new venture. 
    Use the provided knowledge to support your arguments and provide a comprehensive evaluation of the field.

    {{}}

    Rules:
    1. Only use knowledge provided to write.
    2. Back every argument with evidence from the knowledge.
    3. Use in-text citations in APA format.
    4. Critically evaluate strengths and weaknesses of the research theme.
    5. Introduce at least two new critique arguments in each iteration, supported by evidence from the knowledge.
    6. Identify gaps or weaknesses in the previous iteration and enrich them with new insights from the knowledge sources.
    7. Address interdependencies and contextual suitabilities.
    8. Avoid repeating the same argument without elaboration or alternative perspectives.
    9. Explore alternative critiques or perspectives for each argument to enrich the review.
    10. All arguments made in previous iterations must remain in the current iteration, with further enrichment and evidence added where appropriate.
    11. Only write critique arguments on strengths and weaknesses of the field.
    12. Never write an argument without evidence from the knowledge.
    13. Never use cite names of authors in the text, only use in-text citations with APA format.
    14. If there are multiple sources for the same argument, use all of them, aggregated in the same sentence and cite them all using APA format.
    15. Never use the name of the article in the text, only use in-cite citations with APA format.
    16. All references that are available in the knowledge for this iteration should be used and cited in the text.
    17. All of the references available in the previous iteration and arguments made based on them must remain in the current iteration.
    18. The final review should be 2500 words long.
    19. The review should be structured with an introduction, body, and conclusion. Don't add Recommendations for Future Research, Expanding the Review or similat sections.
    20. Ensure each section of the review (introduction, body, and conclusion) is expanded meaningfully to achieve the 2500-word target.

    Example work: 
        Heterogenous defining of social entrepreneurship:
        Firstly, as highlighted in the introduction, scholarly debate on a singular understanding of social entrepreneurship (SE) is unsettled, with the definition varying between fields.
        Huybrehts & Nichols (2012) argue that SE is highly contextual, therefore its understanding varies between the ideologies and goals of the institutions that foster a particular definition.
        This is accurate when contrasted with other scholarly works. For example, in relation to social innovation, SE is understood as the creation of social value through pattern-breaking change or innovation in products, services, organization or production (Phillips et al., 2014).
        Whereas, in relation to opportunity recognition, a process to discover, define and export opportunities in order to create social wealth through new ventures in an innovative way (Yitshaki & Kropp, 2019). In relation to green innovation and sustainable development, Galindo-Martin et al. (2020) view on entrepreneurship is not fully accurate because it sees it as an action-creating process [within an already existing business] and not new venture creation. Nevertheless, their understanding still portrays SE as a process in which social value is seen as a sustainable development created using (green] innovative methods by the entrepreneur. In contrast, when defining SE in relation to effectuation, it is the creation of a venture that has social value creation as a goal, with the means available (Comer & Ho,
        2010).
        The above examples suggest that SE is a diverse and complex matter because it is defined differently based on the context it is established in. Furthermore, the Triple Bottom Line Theory of People, Planet and Profit suggests that social value could be understood in three dimensions as generating value for society, whether by providing them job opportunities, housing and food, environmentally by promoting sustainable actions across all spheres whether in businesses or amongst customers and finally economically [profit], whether it is about donating some of the profits to a charitable cause or investing all finance in the development of SE (Majid & Koe, 2012). Nevertheless, Huybrechts & Nichols (2012) provide an accurate analysis that SE, despite contextual differences, can be based upon three pillars of sociality, innovation and market orientation. This view draws a parallel with the other understandings of SE, as they all share that innovation is a seed of SE and without it, it would not operate effectively. That through innovation social value can be created. Yes, there are limitations to some views that see SE as a process of action-taking, rather than the creation of a new business to take that action, in this case, social value creation (Galindo-Martin et al., 2020). However, most definitions of SE take the primacy of social value creation [sociality] as a determinant of being socially entrepreneurial. Innovation is the catalyst for making that possible and market orientation allows the entrepreneur to be
        focused on up-to-date needs of society, whether, social, economic or environmental. To conclude, the review of this issue suggests that scholars (whether cognitively or not] agree on the fact that SE is determined by the primacy of social value creation through innovation in a particular context, by a new venture and the process of setting up a new venture for this purpose.
    
    References:
        - Drover, W., Busenitz, L., Matusik, S., Townsend, D., Anglin, A., & Dushnitsky, G. (2017). A review and road map of entrepreneurial equity financing research: Venture capital, corporate venture capital, angel investment, crowdfunding, and accelerators. Journal of Management, 43(6), 1820–1853. https://doi.org/10.1177/0149206317690584
        - Ahluwalia, S., Mahto, R. V., & Guerrero, M. (2020). Blockchain technology and startup financing: A transaction cost economics perspective. Technological Forecasting & Social Change, 151, 119854. https://doi.org/10.1016/j.techfore.2019.119854
        - Fisch, C., Meoli, M., & Vismara, S. (2020). Does blockchain technology democratize entrepreneurial finance? An empirical comparison of ICOs, venture capital, and REITs. Technological Forecasting and Social Change, 157, 120099. https://doi.org/10.1016/j.techfore.2020.120099
        - Boakye, E. A., Zhao, H., & Ahia, B. N. K. (2022). Emerging research on blockchainx technology in finance: A conveyed evidence of bibliometric-based evaluations. Journal of High Technology Management Research, 33, 100437. https://doi.org/10.1016/j.hitech.2022.100437
        - Röhm, P. (2018). Exploring the landscape of corporate venture capital: A systematic review of the entrepreneurial and finance literature. Management Review Quarterly, 68(3), 279–319. https://doi.org/10.1007/s11301-018-0140-z
        - Cavallo, A., Ghezzi, A., Dell'Era, C., & Pellizzoni, E. (2019). Fostering digital entrepreneurship from startup to scaleup: The role of venture capital funds and angel groups. Technological Forecasting and Social Change, 145, 24–35. https://doi.org/10.1016/j.techfore.2019.04.022
        - Gutmann, T. (2019). Harmonizing corporate venturing modes: An integrative review and research agenda. Management Review Quarterly, 69(2), 121–157. https://doi.org/10.1007/s11301-018-0148-4
        - Brush, C. G., Edelman, L. F., & Manolova, T. S. (2012). Ready for funding? Entrepreneurial ventures and the pursuit of angel financing. Venture Capital, 14(2-3), 111–129. https://doi.org/10.1080/13691066.2012.654604
        - Svetek, M. (2022). Signaling in the context of early-stage equity financing: Review and directions. Venture Capital, 24(1), 71–104. https://doi.org/10.1080/13691066.2022.2063092
        - Lukkarinen, A., Teich, J. E., Wallenius, H., & Wallenius, J. (2016). Success drivers of online equity crowdfunding campaigns. Decision Support Systems, 87, 26–38. https://doi.org/10.1016/j.dss.2016.04.006
        - Howell, S. T. (2017). Financing innovation: Evidence from R&D grants. American economic review, 107(4), 1136-1164. https://doi.org/10.1257/aer.20150808
        - Hottenrott, H., Lins, E., & Lutz, E. (2017). Public subsidies and new ventures’ use of bank loans. Economics of Innovation and New Technology, 27(8), 786–808. https://doi.org/10.1080/10438599.2017.1408200
        - Plummer, L. A., Allison, T. H., & Connelly, B. L. (2016). Better together? Signaling interactions in new venture pursuit of initial external capital. Academy of Management Journal, 59(5), 1585-1604. https://doi.org/10.5465/amj.2013.0100
        - Chua, J. H., Chrisman, J. J., Kellermanns, F., & Wu, Z. (2011). Family involvement and new venture debt financing. Journal of business venturing, 26(4), 472-488. https://doi.org/10.1016/j.jbusvent.2009.11.002
        - Frid, C. J., Wyman, D. M., Gartner, W. B., & Hechavarria, D. H. (2016). Low-wealth entrepreneurs and access to external financing. International Journal of Entrepreneurial Behavior & Research, 22(4), 531-555. https://doi.org/10.1108/IJEBR-08-2015-0173
        - Wong, A. Y. (2002). Angel finance: the other venture capital. Available at SSRN 941228. http://dx.doi.org/10.2139/ssrn.941228
        - Colombo, O. (2021). The Use of Signals in New-Venture Financing: A Review and Research Agenda. Journal of Management, 47(1), 237-259. https://doi.org/10.1177/0149206320911090
        - Bonini, S., & Capizzi, V. (2019). The role of venture capital in the emerging entrepreneurial finance ecosystem: future threats and opportunities. Venture Capital, 21(2-3), 137-175. https://doi.org/10.1080/13691066.2019.1608697
        - Wallmeroth, J., Wirtz, P., & Groh, A. P. (2018). Venture capital, angel financing, and crowdfunding of entrepreneurial ventures: A literature review. Foundations and Trends® in Entrepreneurship, 14(1), 1-129. http://dx.doi.org/10.1561/0300000066
        - Atherton, A. (2012). Cases of start‐up financing: An analysis of new venture capitalisation structures and patterns. International Journal of Entrepreneurial Behavior & Research, 18(1), 28-47. https://doi.org/10.1108/13552551211201367
        - Stayton, J., & Mangematin, V. (2019). Seed accelerators and the speed of new venture creation. The Journal of Technology Transfer, 44, 1163-1187. https://doi.org/10.1007/s10961-017-9646-0
    
    Article template:
    Introduction (15% of total {word_goal} words)
        - Introduce the topic and its relevance.
        - Provide background context for the review.
        - State the objectives of the review and questions it seeks to address.
        - Outline the structure of the review.
    Body:
        - Critical Analysis (50%  of total {word_goal} words))
            - Exaime each subtopic in detail, including venture capital, corporate venture capital, angel investment, crowdfunding, accelerators, Friends and Family, Bank Loans, grants, Strategic Partnerships, and blockchain technology.
            - Compare studies or perspectives, identifying agreements and disagreements.
            - Critically evaluate methodologies, findings, or interpretations.
            - Discuss unresolved issues, gaps, or controversies in the literature.
            - Assess the contextual relevance and applicability of findings.

        - Synthesis and Discussion (15%  of total {word_goal} words))
            - Integrate insights from the literature into a cohesive understanding.
            - Discuss implications for theory, practice, or policy.
            - Suggest areas for future research or unresolved questions.

    Conclusion (20%  of total {word_goal} words))
        - Recap the main points from the analysis and synthesis.
        - Offer practical or theoretical recommendations based on findings.
        - Reflect on the broader significance of the review.
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

            # Generate the review for the current file
            result = iterative_review_generation(task, qa_chain, result)
            words_written = len(result.split())
            print(f"Iteration {repeat + 1} - file {file_path}: {words_written} words written.\n\n")
        if words_written > word_goal and repeat >= 1:
            break


    # Step 4: Save the result to a text file
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(result)

    print(f"Task completed. Result saved to {output_file_path}")

# Run the script
if __name__ == "__main__":
    main()