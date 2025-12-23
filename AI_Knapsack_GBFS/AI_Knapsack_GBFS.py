from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox
)
import sys
import heapq
import time
import random
def greedy_best_first_search(items, capacity):

    start = time.time()
    steps = 0


    items = sorted(items, key=lambda x: x[0] / x[1], reverse=True)
    n = len(items)


    frontier = []


    h0 = items[0][0] / items[0][1]
    heapq.heappush(frontier, (-h0, 0, 0, 0, []))

    best_value = 0
    best_solution = []

    while frontier:
        steps += 1
        neg_h, idx, cur_w, cur_v, chosen = heapq.heappop(frontier)


        if cur_v > best_value:
            best_value = cur_v
            best_solution = chosen

        if idx >= n:
            continue

        v, w = items[idx]
        if cur_w + w <= capacity:
            next_h = items[idx + 1][0] / items[idx + 1][1] if idx + 1 < n else 0
            heapq.heappush(
                frontier,
                (-next_h, idx + 1, cur_w + w, cur_v + v, chosen + [(v, w)])
            )

        next_h = items[idx + 1][0] / items[idx + 1][1] if idx + 1 < n else 0
        heapq.heappush(
            frontier,
            (-next_h, idx + 1, cur_w, cur_v, chosen)
        )

    elapsed = time.time() - start
    return best_value, best_solution, steps, elapsed


def ant_colony_optimization(items, capacity, n_ants=30, n_iterations=100,
                            alpha=1.0, beta=2.0, rho=0.1, pheromone_init=1.0,
                            use_best_only=True):

    start = time.time()
    n = len(items)
    if n == 0:
        return 0, [], 0, 0.0

    pheromone = [pheromone_init for _ in range(n)]

    heuristic = [(v / w) if w != 0 else float('inf') for v, w in items]


    best_value_global = 0
    best_solution_global = []
    steps = 0



    for iteration in range(n_iterations):
        iteration_solutions = []
        iteration_values = []


        for ant in range(n_ants):
            steps += 1
            remaining = set(range(n))  
            total_w = 0
            total_v = 0
            solution_idx = []

            while True:
                feasible = [i for i in remaining if total_w + items[i][1] <= capacity]
                if not feasible:
                    break


                weights = []
                for i in feasible:
                    tau = pheromone[i] ** alpha
                    eta = heuristic[i] ** beta
                    weights.append(tau * eta)


                weight_sum = sum(weights)
                if weight_sum == 0:
                    chosen_idx = random.choice(feasible)
                else:
                    pick = random.random() * weight_sum
                    cum = 0.0
                    chosen_idx = feasible[-1]  
                    for i_idx, wgt in zip(feasible, weights):
                        cum += wgt
                        if pick <= cum:
                            chosen_idx = i_idx
                            break

                total_w += items[chosen_idx][1]
                total_v += items[chosen_idx][0]
                solution_idx.append(chosen_idx)
                remaining.remove(chosen_idx)


            iteration_solutions.append(solution_idx)
            iteration_values.append(total_v)

            if total_v > best_value_global:
                best_value_global = total_v
                best_solution_global = solution_idx.copy()


        for i in range(n):
            pheromone[i] *= (1 - rho)

            if pheromone[i] < 1e-8:
                pheromone[i] = 1e-8


        if use_best_only:
            best_iter_val = max(iteration_values)
            best_idx = iteration_values.index(best_iter_val)
            best_iter_solution = iteration_solutions[best_idx]


            deposit = best_iter_val / (1.0 + sum(v for v, w in items) / 10.0)
            for i in best_iter_solution:
                pheromone[i] += deposit
        else:
            for sol, val in zip(iteration_solutions, iteration_values):
                deposit = val / (1.0 + sum(v for v, w in items) / 10.0)
                for i in sol:
                    pheromone[i] += deposit


        for i in range(n):
            pheromone[i] += (random.random() * 1e-4)


    best_solution_items = sorted([items[i] for i in best_solution_global], key=lambda x: x[1])
    return best_value_global, best_solution_items, steps, time.time() - start


class KnapsackGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Knapsack - GBFS & Ant Colony (fixed)")
        self.resize(760, 560)


        layout = QVBoxLayout()



        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("Chọn thuật toán:"))
        self.algo_box = QComboBox()
        self.algo_box.addItems([
            "Greedy Best First Search (GBFS)",
            "Ant Colony Optimization (ACO)"
        ])
        algo_layout.addWidget(self.algo_box)
        layout.addLayout(algo_layout)


        cap_layout = QHBoxLayout()
        cap_layout.addWidget(QLabel("Sức chứa (capacity):"))
        self.capacity_input = QLineEdit()
        cap_layout.addWidget(self.capacity_input)
        layout.addLayout(cap_layout)


        layout.addWidget(QLabel("Items (mỗi dòng: value,weight). Ví dụ: 60,10"))
        self.items_input = QTextEdit()
        sample = "60,10"
        self.items_input.setPlainText(sample)
        layout.addWidget(self.items_input)

        aco_layout = QHBoxLayout()
        aco_layout.addWidget(QLabel("ACO: ants"))
        self.ants_input = QLineEdit("30")
        aco_layout.addWidget(self.ants_input)
        aco_layout.addWidget(QLabel("iterations"))
        self.iter_input = QLineEdit("100")
        aco_layout.addWidget(self.iter_input)
        layout.addLayout(aco_layout)

        self.run_btn = QPushButton("Chạy thuật toán")
        self.run_btn.clicked.connect(self.run_algorithm)
        layout.addWidget(self.run_btn)

        layout.addWidget(QLabel("Kết quả:"))
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)


        self.setLayout(layout)

    @staticmethod
    def parse_items(text):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        items = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                raise ValueError("Mỗi dòng phải có 2 số: value,weight")
            v = int(parts[0])
            w = int(parts[1])
            if w < 0 or v < 0:
                raise ValueError("value và weight phải >= 0")
            items.append((v, w))
        return items


    def run_algorithm(self):
        try:
            capacity = int(self.capacity_input.text())
            if capacity < 0:
                raise ValueError
        except ValueError:
            self.output.setText("Capacity phải là số nguyên không âm!")
            return

        try:
            items = self.parse_items(self.items_input.toPlainText())
            if not items:
                raise ValueError("Danh sách items rỗng")
        except Exception as e:
            self.output.setText(f" Lỗi định dạng items: {e}")
            return


        algo = self.algo_box.currentText()

        if algo == "Greedy Best First Search (GBFS)":
            total, sel, steps, t = greedy_best_first_search(items, capacity)
        else:
            try:
                ants = int(self.ants_input.text())
                iters = int(self.iter_input.text())
                if ants <= 0 or iters <= 0:
                    raise ValueError
            except ValueError:
                self.output.setText("ACO params không hợp lệ. ants/iterations phải > 0")
                return

            total, sel, steps, t = ant_colony_optimization(items, capacity,
                                                           n_ants=ants, n_iterations=iters,
                                                           alpha=1.0, beta=2.0, rho=0.1,
                                                           pheromone_init=1.0,
                                                           use_best_only=True)


        # Show output
        result = f"=== {algo} ===\n"
        result += f"Sức chứa: {capacity}\n"
        result += f"Tổng giá trị: {total}\n"
        result += f"Số bước (steps): {steps}\n"
        result += f"Thời gian: {t:.6f} s\n"
        result += "Items được chọn (value, weight):\n"
        for v, w in sel:
            result += f"- ({v}, {w})\n"


        # extra: display total weight if possible
        total_w = sum(w for v, w in sel)
        result += f"Tổng trọng lượng: {total_w}\n"


        self.output.setText(result)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = KnapsackGUI()
    win.show()
    sys.exit(app.exec())




