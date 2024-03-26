import openai
import os
import pandas as pd
from parse_pdf import stem_name, pdf2str, DocumentExtractor
from tqdm import tqdm

API_KEY = 'YOUR_API_KEY'
CLIENT = openai.OpenAI(api_key=API_KEY)

SYS_PROMPT = '''Classify the following 25 ESG tags according to the input paragraph:

Business Ethics: 0
Data Security: 1
Access And Affordability: 2
Business Model Resilience: 3
Competitive Behavior: 4
Critical Incident Risk Management: 5
Customer Welfare: 6
Director Removal: 7
Employee Engagement Inclusion And Diversity: 8
Employee Health And Safety: 9
Human Rights And Community Relations: 10
Labor Practices: 11
Management Of Legal And Regulatory Framework: 12
Physical Impacts Of Climate Change: 13
Product Quality And Safety: 14
Product Design And Lifecycle Management: 15
Selling Practices And Product Labeling: 16
Supply Chain Management: 17
Systemic Risk Management: 18
Waste And Hazardous Materials Management: 19
Water And Wastewater Management: 20
Air Quality: 21
Customer Privacy: 22
Ecological Impacts: 23
Energy Management: 24
GHG Emissions: 25

Output in this format:
[Directly related], [May be related], [Not mentioned]

Make sure that all integers between 0 and 25 exist in exactly one of these lists.'''

# paragraph = "Framework has the potential to enable and mobilize all actors of society to contribute, similar to what the Paris Agreement has meant for limiting climate change"
# [13, 23], [12, 18, 24, 25], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 19, 20, 21, 22]

IN_OUT_TOKENS = [0, 0]
# https://openai.com/pricing
PRICE_GPT4_0125_PRE = 10 / 1_000_000, 30 / 1_000_000  # in, out, USD per tokens
USD_TWD = 32


def ask_gpt_labels(passage: str) -> str:
    completion = CLIENT.chat.completions.create(
        model='gpt-4-0125-preview',
        max_tokens=128,
        temperature=0.,
        messages=[
            {'role': 'system', 'content': SYS_PROMPT},
            {'role': 'user', 'content': passage},
        ]
    )
    IN_OUT_TOKENS[0] += completion.usage.prompt_tokens
    IN_OUT_TOKENS[1] += completion.usage.completion_tokens
    return completion.choices[0].message.content


SAMPLE_PARAGRAPH = [
    "This report summarises our performance compared to  the IKEA sustainability strategy, People & Planet Positive,  during FY22. It covers the entire IKEA value chain and  franchise system and provides an update on activities  and the ongoing work to measure progress. The IKEA  Sustainability Report is issued by Inter IKEA Group",
    "The reporting period follows the financial year 2022  (FY22), which runs from 1 September 2021 to 31 August  2022. Percentages in this report may not total 100% due  to rounding differences",
    "The IKEA business is defined as the business activities  performed by all entities operating under the IKEA  Brand. “We” in this report refers to the IKEA business",
    "One of the main contributors to the decreased  climate footprint is our more energy-efficient range,  such as the new and more affordable SOLHETTA  LED range. We also saw a continued increase in the  share of sales for our plant-based food options, such  as the plant ball and the veggie hotdog",
    "Establishing a systemic shift to a circular economy,  not only within the IKEA business but throughout  the world, is key to creating a sustainable future,  since the high use of resources in the world is  increasing the pressure on people and the planet",
    "Revised the Fair & equal focus area,  strengthening our commitments  to tackle inequality; make respect for  human rights a foundation for business  operations; and contribute to more  resilient societies, including ensuring a just  transition (page 41)"
]


def is_path_creatable(pathname: str) -> bool:
    """
    https://stackoverflow.com/a/34102855
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    """
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dir_name = os.path.dirname(pathname) or os.getcwd()
    return os.access(dir_name, os.W_OK)


class GPTParagraphOptions:
    def __init__(self, file_path_or_paragraph: str | list[str], output_dir, output_name=''):
        """
        :param file_path_or_paragraph:
        csv_path (contains paragraph columns) | pdf_path | paragraph (strings)

        :param output_dir: GPT csv output dir
        :param output_name: specific csv name (must specific if file_path_or_paragraph is paragraph)
        """
        if output_name:
            assert output_name.endswith('.csv')
            self.output_name = output_name
        else:
            self.output_name = stem_name(file_path_or_paragraph) + '_gpt.csv'
        self.output_path = os.path.join(output_dir, self.output_name)

        assert is_path_creatable(self.output_path)

        if type(file_path_or_paragraph) is str:
            if file_path_or_paragraph.endswith('csv'):
                df = pd.read_csv(file_path_or_paragraph, usecols=('paragraph',))
                self.paragraph = df['paragraph'].tolist()
            else:
                document = pdf2str(file_path_or_paragraph)
                extractor = DocumentExtractor(document)
                self.paragraph = extractor.extract()
        else:
            self.paragraph = file_path_or_paragraph


def gpt_labeling(gpt_paragraph_options: GPTParagraphOptions):
    # check output path
    print("Output path:", gpt_paragraph_options.output_path)
    print("Paragraph length:", len(gpt_paragraph_options.paragraph))
    input("Please confirm...")

    paragraph = gpt_paragraph_options.paragraph
    IN_OUT_TOKENS[:] = 0, 0
    gpt_labels = []
    bar = tqdm(paragraph)
    for passage in bar:
        gpt_labels.append(ask_gpt_labels(passage))
        cost = (IN_OUT_TOKENS[0] * PRICE_GPT4_0125_PRE[0] + IN_OUT_TOKENS[1] * PRICE_GPT4_0125_PRE[1]) * USD_TWD
        bar.set_postfix_str("Estimated cost: %d(TWD)" % cost)

    df = pd.DataFrame({'paragraph': paragraph, 'gpt_labels': gpt_labels})
    df.to_csv(gpt_paragraph_options.output_path, index=False)


if __name__ == '__main__':
    # gpt_labeling(GPTParagraphOptions(SAMPLE_PARAGRAPH, '', 'ikea_sample.csv'))
    gpt_labeling(GPTParagraphOptions('ikea_sustainability_report_fy22_57c0217c71_label.csv', ''))
