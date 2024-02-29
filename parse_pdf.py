import pandas as pd
import os
from io import StringIO
from re import compile

# !pip install pdfminer.six
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

DEMO_COLOR = '\33[93m'  # yellow


def stem_name(path: str) -> str:
    """:return: basename without extension"""
    base_name = os.path.basename(path)
    return base_name[: base_name.rfind('.')]


def pdf2str(pdf_path: str) -> str:
    output_string = StringIO()
    with open(pdf_path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
    return output_string.getvalue()


_REG_XML = compile(r'<Font .*?>([^><]+?)</Font>')
_REG_SPACES = compile(r'\s+')


def xml2str(xml_path: str) -> str:
    with open(xml_path, encoding='utf8') as file:
        return '\n'.join(
            _REG_SPACES.sub(' ', s).replace('&#10;', '')
            for s in _REG_XML.findall(file.read())
        )


_REG_FIND_REPEAT = compile(r'(\D+)\1{4,}')  # consecutively repeated (at least 5 times) adjacent substrings


class DocumentExtractor:
    def __init__(self, document: str, is_chinese=None):
        self.IS_CHINESE = self.is_chinese_document_(document) if is_chinese is None else is_chinese
        self.PERIOD = '。' if self.IS_CHINESE else '.'
        self.LINES = document.split('\n')
        self._JOIN = '' if self.IS_CHINESE else ' '
        self._demo = []

    def extract(self) -> list[str]:
        self._demo.clear()
        result, stack = [], []
        for line in self.LINES:
            if not line.strip():  # ignore empty line
                if stack:
                    stack[-1] += '\n'
                else:
                    self._demo.append(line)
                continue
            if 'http' in line or _REG_FIND_REPEAT.search(line):  # bad content
                stack.append(line)
                self._demo.extend(stack)
                stack.clear()
                continue

            meaningful = self.is_meaningful(line, self.IS_CHINESE)
            have_period = line[-10:].count(self.PERIOD) == 1

            if not stack:
                if meaningful:
                    if have_period:
                        result.append(line)
                        self._demo.append(DEMO_COLOR + line + '\33[0m')
                    else:
                        stack.append(line)
                else:
                    self._demo.append(line)
                continue

            # stack_have_period = stack[-1][-10:].count(self.PERIOD) == 1
            if have_period:
                if len(line) > len(stack[-1]) * 1.3:
                    self._demo.extend(stack)
                    stack.clear()
                stack.append(line)
                result.append(self._JOIN.join(stack))
                self._demo.append(DEMO_COLOR + '\n'.join(stack) + '\33[0m')
                stack.clear()
            elif self.are_similar_lengths(stack[-1], line):
                stack.append(line)
            else:
                self._demo.extend(stack)
                stack = [line] if meaningful else []

        self._demo.extend(stack)
        # rstrip period and remove '\n'
        result = [s[:s.rfind(self.PERIOD)].replace('\n', '') for s in result]
        return result

    def demo(self):
        if not self._demo:
            self.extract()
        print('\n'.join(self._demo))

    @staticmethod
    def are_similar_lengths(a: str, b: str) -> bool:
        return 0.7 < len(a) / len(b) < 1.3

    @staticmethod
    def is_chinese_document_(document: str) -> bool:
        c = 0
        bound = min(100, len(document) >> 1)
        for u in map(ord, document[:500]):
            if u >= 19968:
                c += 1
                if c > bound:
                    return True
        return False

    @staticmethod
    def is_meaningful(s: str, is_chinese: bool) -> bool:
        if is_chinese:
            return sum(u >= 19968 for u in map(ord, s)) >= 10
        return sum(len(c) > 1 for c in s.split()) >= 5 or ('.' in s[:-5] and len(s) >= 20)


def file2csv(pdf_or_xml_path: str, output_dir='') -> DocumentExtractor:
    if pdf_or_xml_path.endswith('xml'):
        document = xml2str(pdf_or_xml_path)
    else:
        document = pdf2str(pdf_or_xml_path)
    extractor = DocumentExtractor(document)
    paragraph = extractor.extract()
    df = pd.DataFrame({'paragraph': paragraph})
    csv_path = os.path.join(output_dir, stem_name(pdf_or_xml_path) + '.csv')
    df.to_csv(csv_path, sep=',', index=False, encoding='utf-8')
    return extractor
