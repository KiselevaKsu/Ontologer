import re
import pdfplumber
from typing import Iterator


def extract_text_pages(pdf_path: str) -> Iterator[str]:
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                # 1. Удалить номера страниц (вида 109/1738)
                page_text = re.sub(r'\s*\d+/\d+\s*\n?', ' ', page_text)
                
                # 2. Объединить разорванные строки
                lines = page_text.split('\n')
                merged = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if merged and not re.search(r'[.!?:;]\s*$', merged[-1]):
                        # Добавляем пробел перед склеиванием
                        merged[-1] = merged[-1].rstrip() + ' ' + line
                    else:
                        merged.append(line)
                
                # 3. Финальная очистка: убираем множественные пробелы
                page_text = ' '.join(merged)
                page_text = re.sub(r'\s+', ' ', page_text)
                yield page_text