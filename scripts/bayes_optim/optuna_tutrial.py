# 提供されたソースに基づき、Optunaを用いたベイズ最適化の実装チュートリアルを作成しました。

# これまでの会話で整理した「目的関数」「統計モデル」「獲得関数」といった概念が、実際のコードでどのように記述されるかに注目してください。

# ---

# # Optunaによるベイズ最適化 実装チュートリアル

# Optunaは「Define-by-Run」と呼ばれる柔軟な記法を採用しており、Pythonのループや条件分岐を使って直感的に探索空間を記述できるのが特徴です。

# ## 1. インストールと準備
# まずはOptunaをインストールします。仮想環境（venv）上での実行が推奨されています。

# ```bash
# $ pip install optuna
# ```

# ## 2. 基本的な最適化（目的関数の最小化）
# まずは、数式（目的関数）の最小値を探索する最も基本的なコードです。

# **ポイント**:
# *   `objective(trial)`: これが**目的関数**です。
# *   `trial.suggest_float`: 変数の探索範囲（**探索空間**）を定義します。

# ```python
import optuna

# 1. 目的関数の定義
def objective(trial):
    # 探索空間の定義: x, y を -5 から 5 の範囲で探索
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    
    # 最小化したい数式（例: 2x^2 - 1.05x^4 + ...）
    # ここではソースコードの例を簡略化して記述します
    return (x - 2) ** 2 + (y + 3) ** 2

if __name__ == "__main__":
    # 2. Study（最適化プロセス全体を管理するオブジェクト）の作成
    # direction="minimize" で最小化を指定（デフォルトも最小化）
    study = optuna.create_study(direction="minimize")
    
    # 3. 最適化の実行（100回試行）
    study.optimize(objective, n_trials=100)
    
    # 4. 結果の表示
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
# ```
# ,

# ## 3. 機械学習のハイパーパラメータ最適化
# 次に、機械学習モデル（ここではscikit-learn）のハイパーパラメータを調整する例です。

# **ポイント**:
# *   `suggest_categorical`: モデルの種類（SVMかランダムフォレストか）のようなカテゴリ変数も扱えます。
# *   `suggest_loguniform` (または `log=True`): 学習率や正則化パラメータなど、桁が変わる値の探索に適しています。

# ```python
import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def objective(trial):
    iris = load_iris()
    x, y = iris.data, iris.target

    # 分類器の選択（条件分岐による探索空間の定義）
    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    
    if classifier_name == "SVC":
        # SVMのパラメータ探索 (対数スケールで探索)
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = SVC(C=svc_c, gamma="auto")
    else:
        # ランダムフォレストのパラメータ探索 (整数値)
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=10)

    # 交差検証による精度評価
    score = cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    
    return accuracy # 精度を最大化したい

if __name__ == "__main__":
    # direction="maximize" で最大化（精度など）を指定
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
```
,,

## 4. ガウス過程と獲得関数の利用 (BoTorchSampler)
OptunaのデフォルトはTPE（Tree-structured Parzen Estimator）という手法ですが、**ガウス過程（GP）**と**獲得関数（EIなど）**を明示的に使用したい場合は、`BoTorchSampler`を使用します。これにより、これまでの議論にあった理論的なベイズ最適化が可能になります。

**ポイント**:
*   統計モデルとしてガウス過程回帰が使用されます。
*   獲得関数の最適化（L-BFGS等）が内部で実行されます。

```python
import optuna
from optuna.integration import BoTorchSampler

def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    return x**2 + 2*x # シンプルな関数

if __name__ == "__main__":
    # ガウス過程ベースのサンプラーを指定
    # これにより GP-EI (ガウス過程 + 期待改善量) などが利用可能になる
    sampler = BoTorchSampler()
    
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=50)
# ```
# ,

# ※ さらに高度な使い方として、獲得関数をEIからUCBに変更したり、制約付き最適化を行ったりする場合は、`BoTorchSampler`の引数 `candidates_func` をカスタマイズすることで実現可能です。

# ## 5. 結果の可視化
# 最適化の過程やパラメータの重要度を分析するために、可視化機能が用意されています。

# ```python
# 最適化履歴のプロット（横軸：試行回数、縦軸：目的関数値）
# 収束の様子（Simple Regretの推移）を確認できます
optuna.visualization.plot_optimization_history(study).show()

# パラメータ重要度のプロット
# どの変数が結果に大きく寄与しているかを確認できます
optuna.visualization.plot_param_importances(study).show()
# ```
# ,

# ---

# ### まとめ
# *   **目的関数**: `objective` 関数内にPythonコードで記述。
# *   **統計モデル**: デフォルトはTPEだが、`BoTorchSampler`を使えばガウス過程に変更可能。
# *   **獲得関数**: `BoTorchSampler`内部でEIなどが計算され、その最大化（獲得関数の最適化）が行われる。

# このコードを実行することで、手持ちのPC上でベイズ最適化の挙動を確認することができます。