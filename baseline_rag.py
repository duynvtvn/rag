import logging
import os
import json
import re
import torch
import math
import numpy as np
import faiss
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DPRContextEncoder,
    DPRQuestionEncoder,
    get_constant_schedule
)
from torch.optim import AdamW
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

# Thêm các thư viện cho monitoring
import time
from datetime import datetime, timedelta
import psutil
import GPUtil
import threading

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUMonitor:
    """
    Lớp theo dõi GPU usage và thời gian thực thi
    """

    def __init__(self, log_file='gpu_monitoring.json'):
        """
        Khởi tạo bộ theo dõi GPU

        Args:
            log_file: Đường dẫn file lưu thông tin monitoring
        """
        self.log_file = log_file
        self.monitoring_data = {
            'training': [],
            'evaluation': [],
            'gpu_usage': [],
            'memory_usage': [],
            'timestamps': []
        }
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None

    def start_monitoring(self):
        """Bắt đầu theo dõi GPU trong thread riêng"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Đã bắt đầu theo dõi GPU và tài nguyên hệ thống")

    def _monitor_loop(self):
        """Vòng lặp theo dõi GPU chạy trong thread riêng"""
        while self.monitoring:
            try:
                # Lấy thông tin GPU
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Lấy GPU đầu tiên
                    gpu_usage = gpu.load * 100  # Phần trăm sử dụng GPU
                    gpu_memory = gpu.memoryUsed  # MB đã sử dụng
                    gpu_memory_total = gpu.memoryTotal  # Tổng MB
                    gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100

                    # Lấy thông tin CPU và RAM
                    cpu_percent = psutil.cpu_percent(interval=1)
                    ram_percent = psutil.virtual_memory().percent

                    # Lưu thông tin
                    timestamp = datetime.now().isoformat()
                    elapsed_time = time.time() - self.start_time

                    self.monitoring_data['gpu_usage'].append({
                        'timestamp': timestamp,
                        'elapsed_seconds': elapsed_time,
                        'gpu_usage_percent': gpu_usage,
                        'gpu_memory_mb': gpu_memory,
                        'gpu_memory_percent': gpu_memory_percent,
                        'gpu_memory_total_mb': gpu_memory_total,
                        'cpu_percent': cpu_percent,
                        'ram_percent': ram_percent
                    })

                # Ngủ 5 giây trước khi đo lại
                time.sleep(5)

            except Exception as e:
                logger.warning(f"Lỗi khi theo dõi GPU: {str(e)}")
                time.sleep(5)

    def stop_monitoring(self):
        """Dừng theo dõi GPU"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        logger.info("Đã dừng theo dõi GPU")

    def log_training_metrics(self, epoch, batch, loss, acc, elapsed_time):
        """
        Ghi lại metrics trong quá trình training

        Args:
            epoch: Epoch hiện tại
            batch: Batch hiện tại
            loss: Loss hiện tại
            acc: Accuracy hiện tại
            elapsed_time: Thời gian đã trôi qua
        """
        self.monitoring_data['training'].append({
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'accuracy': acc,
            'elapsed_seconds': elapsed_time
        })

    def log_evaluation_metrics(self, dataset_name, metrics, elapsed_time):
        """
        Ghi lại metrics sau khi evaluation

        Args:
            dataset_name: Tên dataset (seen/unseen)
            metrics: Dictionary các metrics
            elapsed_time: Thời gian evaluation
        """
        self.monitoring_data['evaluation'].append({
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'metrics': metrics,
            'elapsed_seconds': elapsed_time
        })

    def save_monitoring_data(self):
        """Lưu dữ liệu monitoring ra file"""
        # Tính toán thống kê tổng hợp
        if self.monitoring_data['gpu_usage']:
            gpu_usages = [d['gpu_usage_percent'] for d in self.monitoring_data['gpu_usage']]
            gpu_memories = [d['gpu_memory_percent'] for d in self.monitoring_data['gpu_usage']]
            cpu_usages = [d['cpu_percent'] for d in self.monitoring_data['gpu_usage']]
            ram_usages = [d['ram_percent'] for d in self.monitoring_data['gpu_usage']]

            summary = {
                'total_time_seconds': time.time() - self.start_time if self.start_time else 0,
                'total_time_formatted': str(
                    timedelta(seconds=int(time.time() - self.start_time))) if self.start_time else "0:00:00",
                'gpu_usage_avg': np.mean(gpu_usages),
                'gpu_usage_max': np.max(gpu_usages),
                'gpu_usage_min': np.min(gpu_usages),
                'gpu_memory_avg': np.mean(gpu_memories),
                'gpu_memory_max': np.max(gpu_memories),
                'cpu_usage_avg': np.mean(cpu_usages),
                'cpu_usage_max': np.max(cpu_usages),
                'ram_usage_avg': np.mean(ram_usages),
                'ram_usage_max': np.max(ram_usages),
                'total_samples': len(self.monitoring_data['gpu_usage'])
            }

            self.monitoring_data['summary'] = summary

        # Lưu ra file JSON
        with open(self.log_file, 'w') as f:
            json.dump(self.monitoring_data, f, indent=2)

        # Lưu báo cáo text dễ đọc
        report_file = self.log_file.replace('.json', '_report.txt')
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BÁO CÁO THEO DÕI GPU VÀ THỜI GIAN TRAINING\n")
            f.write("=" * 80 + "\n\n")

            if 'summary' in self.monitoring_data:
                s = self.monitoring_data['summary']
                f.write("THỐNG KÊ TỔNG HỢP:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Tổng thời gian: {s['total_time_formatted']} ({s['total_time_seconds']:.2f} giây)\n")
                f.write(
                    f"GPU Usage - Trung bình: {s['gpu_usage_avg']:.2f}%, Max: {s['gpu_usage_max']:.2f}%, Min: {s['gpu_usage_min']:.2f}%\n")
                f.write(f"GPU Memory - Trung bình: {s['gpu_memory_avg']:.2f}%, Max: {s['gpu_memory_max']:.2f}%\n")
                f.write(f"CPU Usage - Trung bình: {s['cpu_usage_avg']:.2f}%, Max: {s['cpu_usage_max']:.2f}%\n")
                f.write(f"RAM Usage - Trung bình: {s['ram_usage_avg']:.2f}%, Max: {s['ram_usage_max']:.2f}%\n")
                f.write(f"Số lần đo: {s['total_samples']}\n\n")

            if self.monitoring_data['training']:
                f.write("TIẾN TRÌNH TRAINING:\n")
                f.write("-" * 40 + "\n")
                for i, data in enumerate(self.monitoring_data['training'][-10:]):  # Chỉ hiển thị 10 mục cuối
                    f.write(
                        f"Epoch {data['epoch']}, Batch {data['batch']}: Loss={data['loss']:.4f}, Acc={data['accuracy']:.4f}, Time={data['elapsed_seconds']:.2f}s\n")
                f.write("\n")

            if self.monitoring_data['evaluation']:
                f.write("KẾT QUẢ EVALUATION:\n")
                f.write("-" * 40 + "\n")
                for data in self.monitoring_data['evaluation']:
                    f.write(f"Dataset: {data['dataset']}\n")
                    f.write(f"Thời gian: {data['elapsed_seconds']:.2f} giây\n")
                    for metric, value in data['metrics'].items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n")

        logger.info(f"Đã lưu dữ liệu monitoring vào {self.log_file} và {report_file}")


# Khởi tạo global monitor
gpu_monitor = GPUMonitor(log_file='/kaggle/working/gpu_monitoring.json')


class BaselineRAGDataset(Dataset):
    """
    Lớp xử lý dữ liệu cho Baseline RAG, đơn giản hóa từ GENKS
    """

    def __init__(self, data, tokenizer, context_len=256, max_length=1024, test=False):
        """
        Khởi tạo lớp xử lý dữ liệu Baseline RAG

        Args:
            data: Dữ liệu đầu vào
            tokenizer: Tokenizer để xử lý văn bản
            context_len: Độ dài tối đa của ngữ cảnh
            max_length: Độ dài tối đa của chuỗi đầu vào
            test: Chế độ kiểm thử
        """
        super(Dataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.max_length = max_length
        self.test = test

    def __getitem__(self, index):
        """
        Lấy mẫu dữ liệu tại vị trí index

        Args:
            index: Vị trí mẫu dữ liệu

        Returns:
            Tuple (input_ids, labels)
        """
        example = self.data[index]

        # Xây dựng đầu vào từ ngữ cảnh đối thoại và tri thức
        input_sequence = self._build_input_sequence(example)

        # Xây dựng đầu ra (nhãn)
        target = example.get('labels', [''])[0]

        # Mã hóa đầu vào và đầu ra
        input_ids = self.tokenizer.encode(input_sequence, truncation=True,
                                          max_length=self.max_length, add_special_tokens=True)
        labels = self.tokenizer.encode(target, truncation=True,
                                       max_length=self.context_len, add_special_tokens=True)

        return torch.tensor(input_ids), torch.tensor(labels)

    def _build_input_sequence(self, example):
        """
        Xây dựng chuỗi đầu vào từ ngữ cảnh và tri thức

        Args:
            example: Mẫu dữ liệu

        Returns:
            Chuỗi đầu vào
        """
        # Xử lý ngữ cảnh đối thoại
        context_parts = []

        # Thêm chủ đề nếu có
        if 'chosen_topic' in example:
            context_parts.append(f"Chủ đề: {example['chosen_topic']}")

        # Thêm lịch sử đối thoại
        role = {'0_Wizard': 'User1', '1_Apprentice': 'User2', '0_Apprentice': 'User2',
                '1_Wizard': 'User1', 0: 'User1', 1: 'User2', 'user1': 'User1', 'user2': 'User2'}

        if 'context' in example:
            for turn in example['context']:
                speaker = role.get(turn.get('speaker', ''), turn.get('speaker', ''))
                text = turn.get('text', '')
                context_parts.append(f"{speaker}: {text}")

        # Kết hợp ngữ cảnh
        context_text = "\n".join(context_parts)

        # Thêm tri thức liên quan
        knowledge_parts = ["Thông tin tham khảo:"]

        # Nếu có tri thức đã kiểm tra, sử dụng nó đầu tiên
        if 'title' in example and example['title'] != 'no_passages_used' and 'checked_sentence' in example:
            knowledge_parts.append(f"[{example['title']}] {example['checked_sentence']}")

        # Thêm tri thức từ knowledge
        if 'knowledge' in example:
            for title, passages in example['knowledge'].items():
                # Thêm tối đa 3 câu từ mỗi đoạn để tránh quá dài
                for i, passage in enumerate(passages[:3]):
                    if not any(passage in p for p in knowledge_parts):  # Tránh trùng lặp
                        knowledge_parts.append(f"[{title}] {passage}")

        # Kết hợp tri thức
        knowledge_text = "\n".join(knowledge_parts)

        # Kết hợp ngữ cảnh và tri thức
        input_sequence = f"{context_text}\n\n{knowledge_text}\n\nPhản hồi:"

        return input_sequence

    def __len__(self):
        """
        Trả về số lượng mẫu dữ liệu

        Returns:
            Số lượng mẫu
        """
        return len(self.data)

    def collate_fn(self, data):
        """
        Hàm gộp batch cho DataLoader

        Args:
            data: Danh sách các mẫu

        Returns:
            Dict batch đã gộp
        """
        from torch.nn.utils.rnn import pad_sequence
        padding_value = self.tokenizer.pad_token_id
        input_ids, labels = zip(*data)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(padding_value),
            'labels': labels,
        }


class BaselineRAG:
    """
    Hệ thống RAG cơ bản không sử dụng GENKS
    Quy trình:
    1. Truy xuất sơ bộ: Sử dụng BM25 hoặc DPR để lấy tập lớn đoạn văn liên quan
    2. Lọc và xếp hạng: Sử dụng mô hình xếp hạng để chọn đoạn văn liên quan nhất
    3. Sinh phản hồi: Sử dụng LM để sinh phản hồi dựa trên các đoạn văn đã chọn
    """

    def __init__(self,
                 model_name='facebook/bart-base',
                 retriever_model_name='facebook/dpr-ctx_encoder-single-nq-base',
                 query_encoder_name='facebook/dpr-question_encoder-single-nq-base',
                 ranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
                 embed_dim=768,
                 top_k_retrieval=20,
                 top_k_rerank=5,
                 retrieval_method='bm25',
                 cache_dir=None):
        """
        Khởi tạo hệ thống RAG cơ bản

        Args:
            model_name: Tên mô hình ngôn ngữ
            retriever_model_name: Tên mô hình bộ truy xuất
            query_encoder_name: Tên mô hình mã hóa truy vấn
            ranker_model_name: Tên mô hình xếp hạng lại
            embed_dim: Chiều của embedding
            top_k_retrieval: Số lượng tài liệu truy xuất
            top_k_rerank: Số lượng tài liệu sau khi xếp hạng lại
            retrieval_method: Phương pháp truy xuất ('bm25' hoặc 'dpr')
            cache_dir: Thư mục cache
        """
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.retrieval_method = retrieval_method.lower()

        # Giai đoạn 1: Bộ truy xuất sơ bộ
        self.initialize_retrievers(retriever_model_name, query_encoder_name, embed_dim)

        # Giai đoạn 2: Bộ xếp hạng lại
        self.initialize_ranker(ranker_model_name)

        # Giai đoạn 3: Sinh phản hồi
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Bộ nhớ đệm tài liệu đã truy xuất
        self.doc_cache = {}

    def initialize_retrievers(self, retriever_model_name, query_encoder_name, embed_dim):
        """
        Khởi tạo các bộ truy xuất: DPR hoặc SBERT + FAISS và BM25

        Args:
            retriever_model_name: Tên mô hình bộ truy xuất
            query_encoder_name: Tên mô hình mã hóa truy vấn
            embed_dim: Chiều của embedding
        """
        # BM25 luôn được khởi tạo để sẵn sàng sử dụng
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_class = BM25Okapi
            self.bm25 = None  # Sẽ được khởi tạo trong build_corpus_index
            self.tokenized_corpus = None
            logger.info("Đã khởi tạo BM25")
        except ImportError:
            logger.warning("Không thể nhập rank_bm25. Cài đặt bằng 'pip install rank-bm25'")
            self.bm25_class = None

        # Nếu method là 'dpr' hoặc DPR có sẵn, khởi tạo DPR
        if self.retrieval_method != 'bm25':
            try:
                self.ctx_encoder = DPRContextEncoder.from_pretrained(retriever_model_name)
                self.query_encoder = DPRQuestionEncoder.from_pretrained(query_encoder_name)
                self.dpr_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
                self.query_tokenizer = AutoTokenizer.from_pretrained(query_encoder_name)
                logger.info("Đã khởi tạo bộ truy xuất DPR")
            except Exception as e:
                logger.info(f"Không thể khởi tạo DPR: {str(e)}, sẽ sử dụng phương pháp thay thế")
                if self.retrieval_method == 'dpr':
                    # Fallback sang SentenceTransformer nếu yêu cầu DPR nhưng không thể tải
                    self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                    self.index = faiss.IndexFlatIP(embed_dim)
                    logger.info("Đã chuyển sang SentenceTransformer")

        # Luôn khởi tạo TF-IDF như backup cuối cùng
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    def initialize_ranker(self, ranker_model_name):
        """
        Khởi tạo bộ xếp hạng lại dựa trên cross-encoder

        Args:
            ranker_model_name: Tên mô hình cross-encoder
        """
        try:
            from sentence_transformers import CrossEncoder
            self.ranker = CrossEncoder(ranker_model_name)
            logger.info(f"Đã khởi tạo bộ xếp hạng CrossEncoder: {ranker_model_name}")
        except:
            logger.info("Không thể khởi tạo CrossEncoder, sẽ sử dụng phương pháp xếp hạng đơn giản")
            self.ranker = None

    def build_corpus_index(self, corpus, cache_path=None):
        """
        Xây dựng index cho corpus để truy xuất nhanh với cả DPR và BM25

        Args:
            corpus: Danh sách các văn bản [(id, text, title)]
            cache_path: Đường dẫn để lưu/tải embeddings từ cache
        """
        self.corpus = corpus
        self.id_to_doc = {item[0]: (item[1], item[2]) for item in corpus}
        texts = [doc[1] for doc in corpus]

        # Luôn xây dựng BM25 vì nó nhẹ và nhanh
        if hasattr(self, 'bm25_class') and self.bm25_class is not None:
            logger.info("Đang xây dựng BM25 index...")
            # Tokenize corpus cho BM25
            self.tokenized_corpus = [text.lower().split() for text in texts]
            self.bm25 = self.bm25_class(self.tokenized_corpus)
            logger.info("Đã xây dựng BM25 index thành công")

        # Nếu phương thức truy xuất không phải BM25, tiếp tục với DPR hoặc phương thức khác
        if self.retrieval_method != 'bm25':
            # Tải embedding từ cache nếu có
            if cache_path and os.path.exists(cache_path):
                logger.info(f"Đang tải embeddings từ cache: {cache_path}")
                self.corpus_embeddings = torch.load(cache_path)
                # Xây dựng FAISS index
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.size(1))
                self.index.add(self.corpus_embeddings.cpu().numpy())
                logger.info("Đã tải và xây dựng index từ cache thành công")
                return

            # Xây dựng index với DPR hoặc SentenceTransformer
            if hasattr(self, 'ctx_encoder') and hasattr(self, 'dpr_tokenizer'):
                # Sử dụng DPR để xây dựng index
                logger.info("Đang tạo embeddings cho corpus với DPR...")
                self.corpus_embeddings = []

                batch_size = 32
                for i in tqdm(range(0, len(texts), batch_size)):
                    batch_texts = texts[i:i + batch_size]
                    with torch.no_grad():
                        inputs = self.dpr_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True,
                                                    max_length=512)
                        embeddings = self.ctx_encoder(**inputs).pooler_output
                    self.corpus_embeddings.append(embeddings)

                self.corpus_embeddings = torch.cat(self.corpus_embeddings, dim=0)

                # Lưu embeddings vào cache nếu có đường dẫn
                if cache_path:
                    logger.info(f"Đang lưu embeddings vào cache: {cache_path}")
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(self.corpus_embeddings, cache_path)

                # Xây dựng FAISS index
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
                self.index.add(self.corpus_embeddings.cpu().numpy())

            elif hasattr(self, 'encoder'):
                # Sử dụng Sentence-Transformer
                logger.info("Đang tạo embeddings cho corpus với SentenceTransformer...")
                texts = [doc[1] for doc in corpus]
                self.corpus_embeddings = self.encoder.encode(texts, show_progress_bar=True, convert_to_tensor=True)
                # Xây dựng FAISS index
                self.index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
                self.index.add(self.corpus_embeddings.cpu().numpy())
            else:
                # Fallback sang TF-IDF
                logger.info("Đang tạo TF-IDF matrix...")
                texts = [doc[1] for doc in corpus]
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def retrieve_documents(self, query, dialogue_history=None, top_k=None):
        """
        Giai đoạn 1: Truy xuất tài liệu sơ bộ, hỗ trợ cả DPR và BM25

        Args:
            query: Câu truy vấn
            dialogue_history: Lịch sử đối thoại (tùy chọn)
            top_k: Số lượng tài liệu cần truy xuất

        Returns:
            List of retrieved documents [(id, text, title, score)]
        """
        if top_k is None:
            top_k = self.top_k_retrieval

        # Kết hợp truy vấn với lịch sử đối thoại nếu có
        if dialogue_history:
            last_n_turns = dialogue_history[-3:]  # Lấy 3 lượt cuối
            combined_query = " ".join(last_n_turns) + " " + query
        else:
            combined_query = query

        # Kiểm tra cache
        if combined_query in self.doc_cache:
            logger.info(f"Sử dụng kết quả từ bộ nhớ đệm cho truy vấn: {combined_query[:50]}...")
            return self.doc_cache[combined_query]

        # Sử dụng BM25 nếu được chọn và đã khởi tạo
        if self.retrieval_method == 'bm25' and hasattr(self, 'bm25') and self.bm25 is not None:
            logger.info("Đang truy xuất với BM25...")
            # Tokenize truy vấn giống cách tokenize corpus
            tokenized_query = combined_query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            indices = np.argsort(-scores)[:top_k]

            # Chuyển indices thành documents
            retrieved_docs = []
            for i, idx in enumerate(indices):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(scores[idx])))

        # Sử dụng DPR hoặc phương pháp thay thế khác
        elif hasattr(self, 'query_encoder'):
            # Sử dụng DPR
            with torch.no_grad():
                inputs = self.query_tokenizer(combined_query, return_tensors="pt")
                query_embedding = self.query_encoder(**inputs).pooler_output.cpu().numpy()

            # Tìm kiếm với FAISS
            scores, indices = self.index.search(query_embedding, top_k)

            # Chuyển indices thành documents
            retrieved_docs = []
            for i, (idx, score) in enumerate(zip(indices.flatten(), scores.flatten())):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(score)))

        elif hasattr(self, 'encoder'):
            # Sử dụng Sentence-Transformer
            query_embedding = self.encoder.encode(combined_query, convert_to_tensor=True)
            # Tìm kiếm với FAISS
            scores, indices = self.index.search(query_embedding.cpu().numpy().reshape(1, -1), top_k)

            # Chuyển indices thành documents
            retrieved_docs = []
            for i, (idx, score) in enumerate(zip(indices.flatten(), scores.flatten())):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(score)))

        else:
            # Fallback sang TF-IDF
            query_vec = self.tfidf_vectorizer.transform([combined_query])
            scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
            indices = np.argsort(-scores)[:top_k]
            scores = scores[indices]

            # Chuyển indices thành documents
            retrieved_docs = []
            for i, (idx, score) in enumerate(zip(indices.flatten(), scores.flatten())):
                doc_id = self.corpus[idx][0]
                doc_text = self.corpus[idx][1]
                doc_title = self.corpus[idx][2]
                retrieved_docs.append((doc_id, doc_text, doc_title, float(score)))

        # Lưu vào cache
        self.doc_cache[combined_query] = retrieved_docs

        return retrieved_docs

    def rerank_documents(self, query, docs, dialogue_history=None, top_k=None):
        """
        Giai đoạn 2: Lọc và xếp hạng lại tài liệu

        Args:
            query: Câu truy vấn
            docs: Danh sách tài liệu từ giai đoạn 1
            dialogue_history: Lịch sử đối thoại (tùy chọn)
            top_k: Số lượng tài liệu cần trả về

        Returns:
            List of reranked documents [(id, text, title, score)]
        """
        if top_k is None:
            top_k = self.top_k_rerank

        # Kết hợp truy vấn với lịch sử đối thoại nếu có
        if dialogue_history:
            # Lấy n câu cuối từ lịch sử đối thoại
            last_n_turns = dialogue_history[-3:]  # Lấy 3 lượt cuối
            combined_query = " ".join(last_n_turns) + " " + query
        else:
            combined_query = query

        if self.ranker:
            # Sử dụng cross-encoder cho việc xếp hạng
            pairs = [(combined_query, doc[1]) for doc in docs]
            scores = self.ranker.predict(pairs)

            # Xếp hạng lại dựa trên điểm số mới
            reranked_docs = [(docs[i][0], docs[i][1], docs[i][2], float(scores[i]))
                             for i in range(len(docs))]
            reranked_docs = sorted(reranked_docs, key=lambda x: x[3], reverse=True)[:top_k]
        else:
            # Đơn giản sắp xếp lại dựa trên điểm truy xuất gốc nếu không có ranker
            reranked_docs = sorted(docs, key=lambda x: x[3], reverse=True)[:top_k]

        return reranked_docs

    def prepare_generation_input(self, query, docs, dialogue_history=None):
        """
        Chuẩn bị đầu vào cho việc sinh phản hồi từ các tài liệu đã xếp hạng

        Args:
            query: Câu truy vấn
            docs: Danh sách tài liệu từ giai đoạn 2
            dialogue_history: Lịch sử đối thoại

        Returns:
            Chuỗi đầu vào cho mô hình sinh
        """
        # Xây dựng lịch sử đối thoại
        context_parts = []

        if dialogue_history:
            for i, utterance in enumerate(dialogue_history):
                speaker = "User1: " if i % 2 == 0 else "User2: "
                context_parts.append(f"{speaker}{utterance}")

        # Thêm truy vấn hiện tại
        current_speaker = "User1: " if len(context_parts) % 2 == 0 else "User2: "
        context_parts.append(f"{current_speaker}{query}")

        context_text = "\n".join(context_parts)

        # Xây dựng phần tri thức
        knowledge_parts = ["Thông tin tham khảo:"]

        for doc_id, doc_text, doc_title, score in docs:
            knowledge_parts.append(f"[{doc_title}] {doc_text}")

        knowledge_text = "\n".join(knowledge_parts)

        # Kết hợp ngữ cảnh và tri thức
        input_text = f"{context_text}\n\n{knowledge_text}\n\nPhản hồi:"

        return input_text

    def generate_response(self, input_text, device='cuda', max_length=128):
        """
        Sinh phản hồi dựa trên đầu vào đã chuẩn bị

        Args:
            input_text: Chuỗi đầu vào đã chuẩn bị
            device: Thiết bị tính toán
            max_length: Độ dài tối đa của phản hồi

        Returns:
            Phản hồi được sinh ra
        """
        # Chuyển sang thiết bị tính toán
        self.model = self.model.to(device)

        # Mã hóa đầu vào
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True,
                                max_length=1024).to(device)

        # Sinh phản hồi
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=4,
                no_repeat_ngram_size=3,
                temperature=0.7
            )

        # Giải mã đầu ra
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def process_query(self, query, dialogue_history=None, device='cuda'):
        """
        Xử lý truy vấn theo quy trình đầy đủ

        Args:
            query: Câu truy vấn
            dialogue_history: Lịch sử đối thoại
            device: Thiết bị tính toán

        Returns:
            Phản hồi cuối cùng và thông tin trung gian
        """
        # Giai đoạn 1: Truy xuất tài liệu sơ bộ
        retrieved_docs = self.retrieve_documents(query, dialogue_history)
        logger.info(f"Giai đoạn 1: Đã truy xuất {len(retrieved_docs)} tài liệu")

        # Giai đoạn 2: Lọc và xếp hạng lại
        reranked_docs = self.rerank_documents(query, retrieved_docs, dialogue_history)
        logger.info(f"Giai đoạn 2: Đã xếp hạng lại lấy {len(reranked_docs)} tài liệu tốt nhất")

        # Chuẩn bị đầu vào cho việc sinh phản hồi
        input_text = self.prepare_generation_input(query, reranked_docs, dialogue_history)

        # Giai đoạn 3: Sinh phản hồi
        response = self.generate_response(input_text, device)
        logger.info(f"Giai đoạn 3: Đã sinh phản hồi: {response[:50]}...")

        return {
            "response": response,
            "retrieved_docs": retrieved_docs[:5],  # Chỉ trả về 5 tài liệu đầu tiên để giảm kích thước
            "reranked_docs": reranked_docs[:3]
        }

    def save(self, path):
        """
        Lưu mô hình

        Args:
            path: Đường dẫn thư mục lưu mô hình
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(os.path.join(path, "baseline_model"))
        self.tokenizer.save_pretrained(os.path.join(path, "baseline_tokenizer"))

    def load(self, path, device='cuda'):
        """
        Tải mô hình

        Args:
            path: Đường dẫn thư mục chứa mô hình
            device: Thiết bị tính toán
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(path, "baseline_model")).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "baseline_tokenizer"))


def evaluate_baseline_rag(model, eval_data, output_file=None, batch_size=16):
    """
    Đánh giá baseline RAG

    Args:
        model: Mô hình BaselineRAG
        eval_data: Dữ liệu đánh giá
        output_file: File lưu kết quả đánh giá
        batch_size: Kích thước batch

    Returns:
        Từ điển kết quả đánh giá bao gồm perplexity (PPL)
    """
    # Bắt đầu đo thời gian evaluation
    eval_start_time = time.time()

    logger.info(f"Đang đánh giá mô hình với {len(eval_data)} mẫu")

    # Chuẩn bị dữ liệu đánh giá
    eval_dataset = BaselineRAGDataset(
        eval_data,
        model.tokenizer,
        context_len=128,
        max_length=512,
        test=True
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        collate_fn=eval_dataset.collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Đặt mô hình ở chế độ đánh giá
    model.model.eval()

    # Chuẩn bị các biến theo dõi
    output_text_collect = []
    true_text_collect = []
    # Biến mới cho tính PPL
    total_loss = 0.0
    total_tokens = 0

    # Tiến hành đánh giá
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            # Chuyển batch lên thiết bị tính toán
            batch = {k: v.cuda() for k, v in batch.items()}

            # Chạy forward pass để tính perplexity
            outputs = model.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            # Tích lũy loss và số lượng token cho tính PPL
            loss = outputs.loss
            # Chỉ tính token thực (không phải padding) trong loss
            non_padding_mask = batch['labels'] != -100
            num_tokens = non_padding_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Sinh phản hồi (giữ nguyên phần này từ code gốc)
            gen_outputs = model.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=3,
                temperature=0.7
            )

            # Giải mã đầu ra
            for i in range(gen_outputs.size(0)):
                output_text = model.tokenizer.decode(gen_outputs[i], skip_special_tokens=True)
                output_text_collect.append(output_text)

            # Lấy nhãn thực
            for i in range(batch['labels'].size(0)):
                label = batch['labels'][i].clone()
                # Thay thế padding token với token thực
                label[label == -100] = model.tokenizer.pad_token_id
                true_text = model.tokenizer.decode(label, skip_special_tokens=True)
                true_text_collect.append(true_text)

    # Tính perplexity
    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
    logger.info(f"Perplexity: {ppl:.4f}")

    # Tính toán các metrics dựa trên phản hồi
    # Chuẩn bị dữ liệu cho đánh giá
    refs = [[ref.lower().split()] for ref in true_text_collect]
    hyps = [hyp.lower().split() for hyp in output_text_collect]

    # Tính BLEU
    smoothie = SmoothingFunction().method1
    bleu1 = sum([sentence_bleu([ref[0]], hyp, weights=(1, 0, 0, 0), smoothing_function=smoothie)
                 for ref, hyp in zip(refs, hyps)]) / len(refs)
    bleu4 = sum([sentence_bleu([ref[0]], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
                 for ref, hyp in zip(refs, hyps)]) / len(refs)

    # Tính ROUGE
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores([' '.join(hyp) for hyp in hyps],
                                        [' '.join(ref[0]) for ref in refs], avg=True)
    except:
        # Fallback nếu có vấn đề với ROUGE
        rouge_scores = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}

    # Tính Knowledge F1 (KF1)
    def f1_score(prediction, ground_truths):
        """Tính F1 giữa dự đoán và ground truth"""
        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truths[0].split()
        common = set(prediction_tokens) & set(ground_truth_tokens)
        num_same = len(common)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    kf1 = sum([f1_score(' '.join(hyp), [' '.join(ref[0])]) for hyp, ref in zip(hyps, refs)]) / len(refs)

    # Tính thời gian evaluation
    eval_elapsed_time = time.time() - eval_start_time

    # Tổng hợp kết quả
    results = {
        'ppl': ppl,  # Thêm perplexity vào kết quả
        'bleu1': bleu1 * 100,
        'bleu4': bleu4 * 100,
        'rouge1': rouge_scores['rouge-1']['f'] * 100,
        'rouge2': rouge_scores['rouge-2']['f'] * 100,
        'rougeL': rouge_scores['rouge-l']['f'] * 100,
        'kf1': kf1 * 100
    }

    # Lưu kết quả đánh giá nếu cần
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Lưu các phản hồi được sinh ra
        with open(output_file.replace('.json', '_responses.txt'), 'w') as f:
            for hyp, ref in zip(output_text_collect, true_text_collect):
                f.write(f"Prediction: {hyp}\n")
                f.write(f"Reference: {ref}\n")
                f.write("-" * 80 + "\n")

    # Log metrics vào GPU monitor nếu có output_file (để xác định dataset name)
    if output_file:
        dataset_name = "seen" if "seen" in output_file else "unseen" if "unseen" in output_file else "unknown"
        gpu_monitor.log_evaluation_metrics(dataset_name, results, eval_elapsed_time)

    return results


def train_baseline_rag(model, train_data, eval_data=None, output_dir='ckpt/baseline_rag',
                       epochs=5, batch_size=8, accumulation_steps=4, learning_rate=2e-5):
    """
    Huấn luyện mô hình baseline RAG

    Args:
        model: Mô hình BaselineRAG
        train_data: Dữ liệu huấn luyện
        eval_data: Dữ liệu đánh giá
        output_dir: Thư mục lưu mô hình
        epochs: Số epochs
        batch_size: Kích thước batch
        accumulation_steps: Số bước tích lũy gradient
        learning_rate: Tốc độ học

    Returns:
        Mô hình đã huấn luyện
    """
    # Bắt đầu monitoring
    gpu_monitor.start_monitoring()
    train_start_time = time.time()

    # Khởi tạo accelerator
    accelerator = Accelerator(gradient_accumulation_steps=accumulation_steps)
    logger.info(f"Đang huấn luyện mô hình với {len(train_data)} mẫu trong {epochs} epochs")

    # Chuẩn bị dữ liệu
    train_dataset = BaselineRAGDataset(
        train_data,
        model.tokenizer,
        context_len=128,
        max_length=512,
        test=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=train_dataset.collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    # Chuẩn bị optimizer và scheduler
    optimizer = AdamW(model.model.parameters(), lr=learning_rate)
    model.model, optimizer, train_dataloader = accelerator.prepare(model.model, optimizer, train_dataloader)
    scheduler = get_constant_schedule(optimizer)
    scheduler = accelerator.prepare(scheduler)

    # Vòng lặp huấn luyện
    for epoch in range(epochs):
        epoch_start_time = time.time()
        accelerator.wait_for_everyone()
        logger.info(f'Epoch {epoch + 1}/{epochs}')

        # Đặt mô hình ở chế độ huấn luyện
        model.model.train()

        tk0 = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []
        acc = []

        for batch_idx, batch in enumerate(tk0):
            batch_start_time = time.time()

            with accelerator.accumulate(model.model):
                output = model.model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # Tính toán độ chính xác
                acc.append((output.logits.argmax(-1) == batch['labels'])[:, 1].float().mean().item())
                losses.append(loss.item())

                # Cập nhật thanh tiến trình
                tk0.set_postfix(loss=sum(losses) / len(losses), acc=sum(acc) / len(acc))
                scheduler.step()

                # Log metrics mỗi 50 batch
                if batch_idx % 50 == 0:
                    elapsed_time = time.time() - train_start_time
                    gpu_monitor.log_training_metrics(
                        epoch=epoch + 1,
                        batch=batch_idx,
                        loss=sum(losses) / len(losses),
                        acc=sum(acc) / len(acc),
                        elapsed_time=elapsed_time
                    )

        # Log thời gian cho epoch
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1} hoàn thành trong {epoch_time:.2f} giây")

        # Lưu mô hình sau mỗi epoch
        os.makedirs(output_dir, exist_ok=True)
        if accelerator.is_local_main_process:
            accelerator.save(accelerator.unwrap_model(model.model).state_dict(), f'{output_dir}/epoch_{epoch}.pt')

        # Đánh giá nếu có dữ liệu đánh giá
        if eval_data:
            results = evaluate_baseline_rag(
                model=model,
                eval_data=eval_data,
                output_file=f'{output_dir}/eval_results_epoch_{epoch}.json'
            )
            logger.info(f"Kết quả đánh giá epoch {epoch}: {results}")

    # Lưu mô hình cuối cùng
    if accelerator.is_local_main_process:
        model.save(output_dir)

    # Dừng monitoring và lưu dữ liệu
    gpu_monitor.stop_monitoring()

    # Tính tổng thời gian training
    total_train_time = time.time() - train_start_time
    logger.info(f"Hoàn thành training trong {str(timedelta(seconds=int(total_train_time)))}")

    # Lưu dữ liệu monitoring
    gpu_monitor.save_monitoring_data()

    return model


def main():
    """
    Hàm chính để chạy quá trình huấn luyện và đánh giá
    """
    # Thiết lập logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Ghi lại thời gian bắt đầu toàn bộ quá trình
    total_start_time = time.time()

    # Khởi tạo mô hình
    logger.info("Đang khởi tạo mô hình BaselineRAG...")
    baseline_rag = BaselineRAG(
        model_name='facebook/bart-base',
        retriever_model_name='facebook/dpr-ctx_encoder-single-nq-base',
        query_encoder_name='facebook/dpr-question_encoder-single-nq-base',
        ranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
        top_k_retrieval=100,
        top_k_rerank=20,
        retrieval_method='bm25'
    )

    # Tải dữ liệu
    logger.info("Đang tải dữ liệu...")
    train_data = json.load(open('/kaggle/input/wizard/train.json'))
    valid_data = json.load(open('/kaggle/input/wizard/valid_seen.json'))
    test_seen_data = json.load(open('/kaggle/input/wizard/test_seen.json'))
    test_unseen_data = json.load(open('/kaggle/input/wizard/test_unseen.json'))

    # Xây dựng corpus
    logger.info("Đang xây dựng corpus...")
    corpus_start_time = time.time()
    corpus = []
    for i, example in enumerate(train_data):
        for title, sentences in example['knowledge'].items():
            for j, sentence in enumerate(sentences):
                doc_id = f"doc_{i}_{title}_{j}"
                corpus.append((doc_id, sentence, title))

    # Xây dựng index cho corpus
    baseline_rag.build_corpus_index(corpus)
    corpus_time = time.time() - corpus_start_time
    logger.info(f"Hoàn thành xây dựng corpus trong {corpus_time:.2f} giây")

    # Huấn luyện mô hình
    logger.info("Bắt đầu huấn luyện mô hình...")
    train_baseline_rag(
        model=baseline_rag,
        train_data=train_data,
        eval_data=valid_data,
        epochs=3,
        batch_size=4,
        accumulation_steps=8,
        output_dir='/kaggle/working/ckpt/baseline_rag'
    )

    # Đánh giá mô hình trên test seen
    logger.info("Đánh giá mô hình trên WoW Seen...")
    eval_seen_start = time.time()
    results_seen = evaluate_baseline_rag(
        model=baseline_rag,
        eval_data=test_seen_data,
        output_file='/kaggle/working/ckpt/baseline_rag/results_seen.json'
    )
    eval_seen_time = time.time() - eval_seen_start

    # Đánh giá mô hình trên test unseen
    logger.info("Đánh giá mô hình trên WoW Unseen...")
    eval_unseen_start = time.time()
    results_unseen = evaluate_baseline_rag(
        model=baseline_rag,
        eval_data=test_unseen_data,
        output_file='/kaggle/working/ckpt/baseline_rag/results_unseen.json'
    )
    eval_unseen_time = time.time() - eval_unseen_start

    # Tính tổng thời gian
    total_time = time.time() - total_start_time

    # Báo cáo kết quả
    logger.info("=" * 80)
    logger.info("KẾT QUẢ CUỐI CÙNG")
    logger.info("=" * 80)

    logger.info("\nKết quả trên WoW Seen:")
    metrics_order = ['ppl', 'bleu1', 'bleu4', 'rouge1', 'rouge2', 'rougeL', 'kf1']
    for metric in metrics_order:
        if metric in results_seen:
            logger.info(f"{metric}: {results_seen[metric]:.2f}")

    logger.info("\nKết quả trên WoW Unseen:")
    for metric in metrics_order:
        if metric in results_unseen:
            logger.info(f"{metric}: {results_unseen[metric]:.2f}")

    logger.info("\n" + "=" * 80)
    logger.info("THỜI GIAN THỰC THI")
    logger.info("=" * 80)
    logger.info(f"Xây dựng corpus: {corpus_time:.2f} giây")
    logger.info(f"Đánh giá WoW Seen: {eval_seen_time:.2f} giây")
    logger.info(f"Đánh giá WoW Unseen: {eval_unseen_time:.2f} giây")
    logger.info(f"Tổng thời gian: {str(timedelta(seconds=int(total_time)))}")

    # Lưu báo cáo tổng hợp cuối cùng
    final_report = {
        'results_seen': results_seen,
        'results_unseen': results_unseen,
        'timing': {
            'corpus_build_seconds': corpus_time,
            'eval_seen_seconds': eval_seen_time,
            'eval_unseen_seconds': eval_unseen_time,
            'total_seconds': total_time,
            'total_formatted': str(timedelta(seconds=int(total_time)))
        },
        'timestamp': datetime.now().isoformat()
    }

    with open('/kaggle/working/ckpt/baseline_rag/final_report.json', 'w') as f:
        json.dump(final_report, f, indent=2)

    logger.info("\nĐã lưu báo cáo cuối cùng vào final_report.json")
    logger.info("Hoàn thành toàn bộ quy trình!")


if __name__ == '__main__':
    main()