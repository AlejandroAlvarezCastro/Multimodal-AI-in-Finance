class PromptBuilder:
    @staticmethod
    def prompt_qa(text: str) -> list:
        return [
            {
                'role': 'system',
                'content': """
                You are a model designed to classify interventions in meetings or conferences into three categories:
                [Question]: If the intervention has an interrogative tone or seeks information.
                [Answer]: If the intervention provides information or responds to a previous question.
                [Procedure]: If the intervention is part of the meeting protocol, such as acknowledgments, moderation steps, or phrases without substantial informational content.
                """
            },
            {
                'role': 'user',
                'content': f"""
                Here is the text of the intervention: "{text}"
                """
            }
        ]

    @staticmethod
    def prompt_10k(text: str) -> list:
        return [
            {
                'role': 'system',
                'content': 
                """
                You are an expert in financial reporting and the structure of the SEC Form 10-K. Your task is to analyze a given text excerpt from an earnings call or conference 
                where an S&P 500 company presents its financial and operational updates. You must classify it into one of the following sections of the 10-K report, 
                ensuring there is no overlap or ambiguity.

                [Business]: Describes the company's core operations, including its main products or services, target markets, competitive positioning, industry landscape, 
                corporate structure, and long-term strategy. Discussions about mergers, acquisitions, partnerships, and market expansions fall under this category. 
                Do not include financial performance, risk analysis, or management interpretations.

                [Risk Factors]: Identifies significant risks that could negatively impact the company, such as regulatory challenges, macroeconomic factors, 
                supply chain disruptions, cybersecurity threats, litigation, or market competition. This section is strictly focused on identifying potential risks, 
                not discussing actual financial performance or strategic decisions.

                [MD&A]: Contains management's discussion and analysis of financial results, explaining trends, operational performance, revenue drivers, cost structures, 
                and forward-looking expectations. This section provides context for financial data, identifying why certain numbers changed and how the company is responding to 
                economic conditions.

                [Other]: If the text does not fit into any of the categories above, classify it here.
                """
            },
            {
                'role': 'user',
                'content': f"""    
                Here is the text of the intervention: "{text}"
                """
            }
        ]
    
                # [Financial Statements and Supplementary Data]: Includes official, audited financial statements such as the income statement, balance sheet, and cash flow statement.
                # It also contains detailed accounting disclosures, footnotes, and regulatory reporting requirements. This section presents raw financial figures without 
                # interpretation or strategic commentary.
        
                # [Selected Financial Data]: Provides a high-level summary of key financial metrics over past fiscal years, such as revenue, net income, earnings per share (EPS), 
                # and return on assets (ROA). This section presents raw numerical data but does not provide explanations or insights into financial trends.

    @staticmethod
    def explain_why_other(text: str) -> list:
        return [
            {
                'role': 'system',
                'content': 
                """
                You are an expert in financial reporting and the structure of the SEC Form 10-K. 
                Previously, a text excerpt was analyzed to classify it into one of the following categories:
                
                [Business]: Outlines the company's operations, including its main products/services, target markets, market presence, and strategic objectives.
                [Risk Factors]: Identifies significant risks (market, regulatory, financial) that could adversely affect the company's stability or performance.
                [Selected Financial Data]: Highlights key financial metrics (revenue, net income, assets) over recent years, offering a high-level financial snapshot.
                [MD&A]: Analyzes financial results, explaining trends, challenges, and strategies, and provides management's perspective on future performance.
                [Financial Statements and Supplementary Data]: Contains audited financial statements (income statement, balance sheet, cash flow) and detailed notes adhering to accounting standards. 
                
                The excerpt was classified as [Other], meaning it did not fit into any of the above categories. 

                Your task is to carefully analyze the text again and provide a detailed explanation of why it does not align with any of these categories. 
                """
            },
            {
                'role': 'user',
                'content': f"""
                Here is the text of the intervention: "{text}"
                """
            }
        ]
    
    @staticmethod
    def analize_qa(intervention: str, response: str) -> list:
        return [
            {
                "role": "system",
                "content": """
                              You are an assistant that analyzes multi-question interventions from business or financial conference calls.

                              Given an intervention that may contain several questions, and the subsequent response:
                              - Extract each individual question clearly.
                              - Determine whether each question is answered ("yes"), partially answered ("partially"), or not answered ("no").
                              - If it is answered (fully or partially), provide a short summary and a quote from the response that supports your evaluation.

                              Return the output in a json format as specified
                              """
                                          },
                                          {
                "role": "user",
                "content": f"""
                              Intervention:
                              {intervention}

                              Response:
                              {response}
                """
            }
        ]

    @staticmethod
    def prompt_monologue(text: str) -> list:
        return [
                {
                    'role': 'system',
                    'content': """
                    You are a model designed to classify interventions in meetings or conferences into three categories:
                    [Monologue]: This refers to substantial speeches or statements by company executives, such as CEOs or CFOs, that provide insights into the company's performance, strategic direction, 
                                financial results, future plans, or market outlook. These interventions often include data, analysis, reflection, or forward-looking statements. 
                                They may refer to quarterly earnings, business units, cost structures, investments, partnerships, market risks, or regulatory aspects. hey are usually long interventions with many phrases.
                    [Procedure]: If the intervention is part of the meeting protocol, such as acknowledgments, moderation steps, or phrases without substantial informational content.
                    """
                },
                {
                    'role': 'user',
                    'content': f"""
                    Here is the text of the intervention: "{text}"
                    """
                }
            ]
    
    @staticmethod
    def check_coherence(monologue: str, response: str) -> list:
        return [
            {
                "role": "system",
                "content": """
    You are an expert assistant in business communication analysis and logical consistency verification.

    Your task is to assess whether a *response* from a company executive is related to and logically aligned with a prior *monologue* (e.g., during an earnings call or public statement).

    Follow these steps:
    1. Check if the **topic** addressed in the response is mentioned or discussed in the monologue.
    2. Determine if the response is **logically consistent** with the content of the monologue.
    3. If there are any inconsistencies or contradictions, provide:
    - A relevant excerpt from the monologue.
    - A relevant excerpt from the response.
    - A brief explanation of the contradiction.
    4. If the response is inconsistent, the `contradictions` list must contain at least one object.

    Respond in **strict JSON format** as follows:
    {
    "topic_covered": true | false,  // Is the topic of the response present in the monologue?
    "consistent": true | false,     // Does the response align with the monologue?
    "summary": "Brief explanation of the consistency or contradiction",
    "contradictions": [             // List of specific contradictions, if any
        {
        "monologue_excerpt": "...",
        "response_excerpt": "...",
        "explanation": "..."
        },
        ...
    ]
    }
                """
            },
            {
                "role": "user",
                "content": f"""
    Monologue:
    {monologue}

    Response:
    {response}
                """
            }
        ]