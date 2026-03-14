import numpy as np
from typing import List
from scipy.ndimage import gaussian_filter
from sklearn.metrics.pairwise import cosine_similarity

import logging.config

logger = logging.getLogger(__name__)

def apply_kernel_method(self, retrieved_indices: List[int], sigma=4.0, threshold=0.13) -> List[int]:
    """
    Расширение контекста, основанное на гауссовском фильтре.
    """

    logger.info(f"Начальные чанки: {retrieved_indices}")
    logger.info(f"Начальные чанки (отсортированные): {sorted(retrieved_indices)}")

    pages_map = self.pages_map

    relevant_page_ids = set()

    for idx in retrieved_indices:
        meta = self.metadatas[idx]
        pid = meta['page_id']
        relevant_page_ids.add(pid)

    final_indices = retrieved_indices

    for pid in relevant_page_ids:
        page_chunks = pages_map[pid]
        if not page_chunks:
            continue

        max_order = max(c[0] for c in page_chunks)
        timeline = np.zeros(max_order + 1)

        retrieved_set = set(retrieved_indices)
        for order, orig_idx in page_chunks:
            if orig_idx in retrieved_set:
                timeline[order] = 1.0

        density = gaussian_filter(timeline, sigma=sigma)

        passing_orders = np.where(density > threshold)[0]
        passing_set = set(passing_orders)

        for order, orig_idx in page_chunks:
            if order in passing_set:
                final_indices.append(orig_idx)

    total_indices = sorted(list(set(final_indices)))

    logger.info(f"Итоговые чанки: {total_indices}")

    return total_indices


def apply_dynamic_expansion(self, retrieved_indices: List[int], base_threshold: float = 0.6,
                            penalty_factor: float = 0.05, max_steps: int = 3) -> List[int]:
    """
    Динамическое расширение контекста на основе семантической близости соседей.

    Args:
        retrieved_indices: Список индексов изначально найденных чанков.
        base_threshold: Базовый порог косинусного сходства для добавления соседа.
        penalty_factor: На сколько увеличивается порог (или уменьшается скор) с каждым шагом удаления.
        max_steps: Максимальное количество шагов влево/вправо от оригинального чанка.
    """
    logger.info(f"Запуск динамического расширения. Исходные: {retrieved_indices}")

    final_indices = set(retrieved_indices)
    processed_indices = set(retrieved_indices)  # Чтобы не обрабатывать одни и те же чанки дважды

    # Создаем карту {pid: {order: global_idx}} для быстрого доступа по O(1)
    pid_order_map = {}
    for pid, chunk_list in self.pages_map.items():
        pid_order_map[pid] = {order: idx for order, idx in chunk_list}

    for idx in retrieved_indices:
        meta = self.metadatas[idx]
        pid = meta.get('page_id')
        current_order = meta.get('chunk_order')

        if pid is None or current_order is None:
            continue

        # Эмбеддинг центрального чанка (anchor)
        anchor_emb = self.embeddings[idx].reshape(1, -1)

        # Проверяем соседей в двух направлениях: -1 (влево) и +1 (вправо)
        for direction in [-1, 1]:
            for step in range(1, max_steps + 1):
                target_order = int(current_order) + (step * direction)

                # Проверяем, существует ли сосед с таким порядковым номером на этой странице
                if pid in pid_order_map and target_order in pid_order_map[pid]:
                    neighbor_idx = pid_order_map[pid][target_order]

                    # Если сосед уже в списке выборки, идем дальше, но не прерываем цепочку
                    # (возможно, следующий за ним тоже релевантен, хотя это редкость)
                    if neighbor_idx in processed_indices:
                        continue

                    # Считаем близость
                    neighbor_emb = self.embeddings[neighbor_idx].reshape(1, -1)
                    similarity = cosine_similarity(anchor_emb, neighbor_emb)[0][0]

                    # Динамический порог: чем дальше от центра, тем строже требование
                    # Пример: порог 0.6, шаг 1 -> надо побить 0.6. Шаг 2 -> надо побить 0.6 + 0.05 = 0.65
                    current_threshold = base_threshold + (penalty_factor * (step - 1))

                    if similarity >= current_threshold:
                        final_indices.add(neighbor_idx)
                        logger.debug(f"Добавлен сосед {neighbor_idx} (sim={similarity:.4f} >= {current_threshold})")
                    else:
                        # Если цепочка прервалась (сосед не подошел), дальше в этом направлении не идем
                        break
                else:
                    # Если чанка с таким номером нет (конец/начало документа), прерываем
                    break

    total_indices = sorted(list(final_indices))
    logger.info(f"Итоговые чанки после динамического расширения: {total_indices}")
    return total_indices