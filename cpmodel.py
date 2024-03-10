from datasets import load_dataset
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import re
import torch
import pandas as pd

PROS_TEMPLATE_PROMPT = """You have been provided with a user review of the product. Please write down the benefits that you found in the review.
The answer should contain no more than 2-3 words of merit.
If you have not found any advantages, then do not try to come up with them, just return only the word "No".
Write the pros separated by commas: "word_1, word_2, word_3..."

Example 1:
Review: "Excellent value for money, well-made product, perfect for our coffee pot, for a small fraction of the cost of the same product in stores."
You answer (word_1, word_2, word_3...): "Good quality, competitive price"

Example 2:
"There is a terrible one band across the top. Supposed to keep the shirt on your shoulders but seems more of like a waistband."
You answer (word_1, word_2, word_3... or No): "No"

Your assessment steps:
1. Read the user review thoroughly.
2. Identify the key words or phrases that suggest benefits.
3. Extract those benefits and write them as single words.
4. Separate the benefits with commas: "word_1, word_2, word_3..."


Review:{review}

You answer (word_1, word_2, word_3...or No):"""

CONS_TEMPLATE_PROMPT = """You have been provided with a user review of the product. Please describe the cons that you found in the review.
The answer should contain no more than 2-3 words that describe the construction.
If you have not found any cons, then do not try to come up with them, just return only the word "No"
Write the pros separated by commas: "word_1, word_2, word_3..."
Look at an example to better understand your task.

Example 1:
"There is a terrible one band across the top. Supposed to keep the shirt on your shoulders but seems more of like a waistband."
You answer (word_1, word_2, word_3... or No): "Terrible streak, not the right size"

Example 2:
Review: "Excellent value for money, well-made product, perfect for our coffee pot, for a small fraction of the cost of the same product in stores."
You answer (word_1, word_2, word_3...): "No"

Your assessment steps:
1. Read the user review carefully.
2. Identify any cons related to construction mentioned.
3. If no cons are mentioned, return "No".
4. If cons are mentioned, write down the keywords that describe them.
5. Return the answer with the cons keywords.


Review:{review}

You answer (word_1, word_2, word_3...or No):"""

SORT_TEMPLATE_PROMPT = """1. You will be given the words, please divide them into pros and cons. 
2. Don't add unnecessary words.
3. If there are no pros, then write in Answer "Pros: No".
3. If there are no cons, then write in Answer "Cons: No".

Example 1:
Words: 'good sound quality, affordable, includes pouch and cable, terrible quality'
Answer: 'Pros: good sound quality, affordable, includes pouch and cable', 'Cons: terrible quality'

Example 2:
Words: 'powerful, heavy, longer cord, suction, crevasse tool, , shorter cloth bag, harder to clean'
Answer: 'Pros: powerful, longer cord, includes suction and crevasse tool', 'Cons: heavy, shorter cloth bag, harder to clean'

Example 3:
Words: 'comfortable, attractive, offers support, variety'
Answer: 'Pros: comfortable, attractive, offers support, variety', 'Cons: No'

Example 4:
Words: 'defective product, unsatisfied customer, broken legs, manufacturer defect'
Answer: 'Pros: defective product, unsatisfied customer, broken legs, manufacturer defect', 'Cons: defective product'


Words: '{pros_cons}'
Answer:"""


class CPModel:
    """
    Класс для суммаризации отзывов на товары с целью извлечения плюсов и минусов.
    
    Параметры:
    - model_name: str - имя или путь к предварительно обученной модели трансформера.
    
    Атрибуты:
    - cp_temperature: float - температура для генерации плюсов и минусов.
    - sort_temperature: float - температура для сортировки плюсов и минусов.
    - cp_max_new_tokens: int - максимальное количество новых токенов для генерации плюсов и минусов.
    - sort_max_new_tokens: int - максимальное количество новых токенов для сортировки плюсов и минусов.
    - tokenizer: AutoTokenizer - токенизатор из библиотеки transformers.
    - model: AutoModelForCausalLM - модель для каузального языкового моделирования из библиотеки transformers.
    """

    def __init__(self, model_name: str):
        """
        Инициализирует модель с заданными параметрами.
        """

        self.cp_temperature = 0.7
        self.sort_temperature = 0.2

        self.cp_max_new_tokens = 20
        self.sort_max_new_tokens = 50
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype = torch.float16)
    
    def get_generation_model(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        """
        Генерирует текст на основе заданного запроса.
        
        Параметры:
        - prompt: str - текст запроса для генерации.
        - temperature: float - температура для генерации текста.
        - max_new_tokens: int - максимальное количество новых токенов для генерации.
        
        Возвращает:
        - Строка сгенерированного текста.
        """

        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs, pad_token_id=2 ,temperature=temperature, max_new_tokens=max_new_tokens, do_sample=True)
        gen_output = self.tokenizer.decode(outputs[0, len(inputs[0]):], skip_special_tokens=True)
        return gen_output
    
    def get_generation_cons(self, review: str) -> list[str]:
        """
        Генерирует список минусов из отзыва.
        
        Параметры:
        - review: str - текст отзыва.
        
        Возвращает:
        - Список строк с минусами.
        """

        prompt = CONS_TEMPLATE_PROMPT.format(review=review)
        gen_output = self.get_generation_model(prompt, temperature=self.cp_temperature, max_new_tokens=self.cp_max_new_tokens)
        extract_get = self.extract_text(gen_output)
        return extract_get

    def get_generation_pros(self, review: str) -> list[str]:
        """
        Генерирует список плюсов из отзыва.
        
        Параметры:
        - review: str - текст отзыва.
        
        Возвращает:
        - Список строк с плюсами.
        """

        prompt = PROS_TEMPLATE_PROMPT.format(review=review)
        gen_output = self.get_generation_model(prompt, temperature=self.cp_temperature, max_new_tokens=self.cp_max_new_tokens)
        extract_get = self.extract_text(gen_output)
        return extract_get
    
    def get_generation_sort_cp(self, pros: list, cons: list) -> dict:
        """
        Сортирует и объединяет плюсы и минусы в один словарь.
        
        Параметры:
        - pros: list - список плюсов.
        - cons: list - список минусов.
        
        Возвращает:
        - Словарь с ключами 'pros' и 'cons', содержащий отсортированные списки плюсов и минусов.
        """

        proc_cons = pros + cons
        proc_cons = list(dict.fromkeys(proc_cons))
        proc_cons = ", ".join(proc_cons)
        
        proc_cons_prompt = SORT_TEMPLATE_PROMPT.format(pros_cons=proc_cons)  
        gen_output = self.get_generation_model(proc_cons_prompt, temperature=self.sort_temperature, max_new_tokens=self.sort_max_new_tokens)
        dict_gen = self.parse_reviews(gen_output)
        return dict_gen
        
    def predict(self, reviews: list[str]) -> list[dict[str]]:
        """
        Производит предсказание плюсов и минусов для списка отзывов.
        
        Параметры:
        - reviews: list[str] - список текстов отзывов.
        
        Возвращает:
        - Список словарей с ключами 'pros', 'cons' и 'review', содержащих информацию о плюсах, минусах и самом отзыве.
        """

        gen_lst = []
        for review in reviews:
            cons = self.get_generation_cons(review)
            pros = self.get_generation_pros(review)
            dict_gen = self.get_generation_sort_cp(pros=pros, cons=cons)
            dict_gen['review'] = review
            gen_lst.append(dict_gen)
        return gen_lst
    
    def extract_text(self, text: str) -> list[str]:
        """
        Очищает и обрабатывает текст для дальнейшего использования.
        
        Параметры:
        - text: str - текст для обработки.
        
        Возвращает:
        - Список строк после очистки и обработки текста.
        """

        text = re.sub(r'\n.*', '', text)
        text = re.sub(r'\s*\([^)]*\)\s*', '', text)
        text = re.sub(r'\[.*', '', text)
        text = text.replace('"', '')
        text = text.replace('You answer', '')
        text = text.replace('\n', '')
        text = text.replace('Review', '')
        text = text.replace('(', '')
        text = text.replace(')', '')
        text = text.replace('.', '')
        lst_text = [item.strip().lower() for item in text.split(',')]
        if len(lst_text) == lst_text.count('no'):
            return ['']
        else:
            lst_text = [item for item in lst_text if item != 'no']
        lst_text = list(dict.fromkeys(lst_text))

        return lst_text
    
    def parse_reviews(self, input_str: str) -> dict[str]:
        """
        Разбирает строку на плюсы и минусы и возвращает их в виде словаря.

        Параметры:
        - input_str: str - строка для разбора.

        Возвращает:
        - Словарь с ключами 'pros' и 'cons', содержащий списки плюсов и минусов соответственно.
        """

        # Приведение всего текста к нижнему регистру и удаление нежелательных символов
        input_str = input_str.lower().replace("'", "").replace(".", "")

        # Извлечение разделов "pros" и "cons"
        pros_text = cons_text = ''
        if "pros:" in input_str and "cons:" in input_str:
            if input_str.find("pros:") < input_str.find("cons:"):
                pros_text, cons_text = re.split("cons:", input_str, 1)
            else:
                cons_text, pros_text = re.split("pros:", input_str, 1)
        pros_text = pros_text.replace("pros:", "")
        cons_text = cons_text.replace("cons:", "")

        # Очистка и разделение на элементы
        clean_pros = [item.strip().capitalize() for item in pros_text.split(",") if item.strip()]
        clean_cons = [item.strip().capitalize() for item in cons_text.split(",") if item.strip()]

        # Обработка случая, когда список пуст
        if not clean_pros:
            clean_pros.append('No')
        if not clean_cons:
            clean_cons.append('No')

        return {"pros": clean_pros, "cons": clean_cons}