import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
import optuna as opt 
import numpy as np
from itertools import permutations


class SearchBestModel:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    y_train: list
    X_test: pd.DataFrame
    y_test: list
    group_train: list
    group_test: list
    features: list
    params: dict
    index_total: int

    def set_study_data(self, df:pd.DataFrame, features, test_size=0.2):
        """ 学習用データを用意 """
        # 訓練データと検証データをランダムに分割（test_size=0.2の場合、グループ数の2割を検証用に指定）
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df['race_id']))
        self.train_df:pd.DataFrame = df.iloc[train_idx]
        self.test_df:pd.DataFrame = df.iloc[test_idx]

        # 特徴量(X) と 正解ラベル(y)
        self.X_train, self.y_train = self.train_df[features], self.train_df["relevance"]
        self.X_test, self.y_test = self.test_df[features], self.test_df["relevance"]

        # グループ情報の作成（各レースに含まれる艇数の配列）
        self.group_train = self.train_df.groupby("race_id").size().values
        self.group_test = self.test_df.groupby("race_id").size().values

        # 特徴量
        self.features = features

        # モデルパラメーター
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 2, 3],
            'random_state': 42,
            'verbose': -1
        }

    def create_study_model(self):
        """ 学習モデルを作成 """
        # モデルの型を定義
        model = lgb.LGBMRanker(**self.params)

        # group パラメータを渡して学習
        model.fit(
            self.X_train, self.y_train,
            group=self.group_train,
            eval_set=[(self.X_test, self.y_test)],
            eval_group=[self.group_test],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        return model

    def get_hit_index_total(self, df:pd.DataFrame):
        """ 120通りの予測から的中したインデックスの合計を取得 """
        index_total = 0
        for _, group in df.groupby("race_id"):
            # Softmax関数。それぞれが0-1の間の数値となり、合計が1となるよう調整
            scores = group["forecast"].values
            exp_s = np.exp(scores - np.max(scores))
            frame_prob = exp_s / np.sum(exp_s)
            # 6艇のすべての組み合わせ(120通り)の確率を取得
            probs = {}
            for i, j, k in permutations(range(len(scores)), 3):
                p1 = frame_prob[i]
                p2 = frame_prob[j] / (1.0 - p1)
                p3 = frame_prob[k] / (1.0 - p1 - p2)
                combo_name = f"{i+1}-{j+1}-{k+1}"
                probs[combo_name] = p1 * p2 * p3
            # 確率が高い順に並び替え
            probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

            # 実際の3連単の結果を取得
            actual_sorted = group.sort_values("rank", ascending=True)
            actual_top3 = list(actual_sorted.iloc[:3]["frame"])
            actual_top3 = "-".join([str(i) for i in actual_top3])

            # 実際の3連単の結果が120通りの何番目に来るかのインデックスを取得
            index = list(probs.keys()).index(actual_top3)
            # トータルに加算
            index_total += index
        return index_total

    def search_best_features(self, update=True):
        """ 最適な特徴量を探索 """
        # 元の学習データを保管
        default_X_train = self.X_train.copy()
        default_X_test = self.X_test.copy()
        default_features = self.features.copy()

        # 探索
        current_features = self.features.copy()
        best_features = []
        best_index_total = None
        removed_feature = ""
        # 特徴量が1つになるまで貢献度が低いものを削る
        while len(current_features) > 0:
            # 学習モデル作成
            self.X_train = self.X_train[current_features]
            self.X_test = self.X_test[current_features]
            model = self.create_study_model()

            # テストデータでの精度評価
            self.test_df["forecast"] = model.predict(self.test_df[current_features])
            # 120通りの予測から的中したインデックスの合計を取得
            index_total = self.get_hit_index_total(self.test_df)
            if best_index_total is None:
                best_index_total = index_total
            # 合計インデックスが最適解よりもいいものであれば、特徴量と合計インデックスを更新
            if index_total <= best_index_total:
                best_features = current_features.copy()
                best_index_total = index_total
            print(f"現在の合計インデックス: {index_total} | 削除された特徴量: {removed_feature}")
            # Gain（貢献度）が最も低い特徴量を特定して削除
            importances = model.booster_.feature_importance(importance_type='gain')
            min_imp_idx = np.argmin(importances)
            removed_feature = current_features.pop(min_imp_idx)
        print(f"最適な特徴量リスト: {best_features} | 合計インデックス: {best_index_total}")

        # 最適な学習データに更新
        if update:
            self.X_train = default_X_train[best_features]
            self.X_test = default_X_test[best_features]
            self.features = best_features
        else: # 元の状態に戻す
            self.X_train = default_X_train
            self.X_test = default_X_test
            self.features = default_features
        self.index_total = best_index_total

    def objective(self, trial:opt.Trial):
        """ Optunaの目的関数 (objective) の定義 """
        # パラメータの探索範囲を設定
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 2, 3],
            'random_state': 42,
            'verbose': -1,  # 学習中のログ出力を抑制
            
            # チューニング対象パラメータ
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        # モデルの定義と学習
        model = self.create_study_model()
        
        # 検証データでの評価スコア（ベストなNDCG@3スコア）を目的関数として返す
        best_score = model.best_score_['valid_0']['ndcg@3']
        return best_score

    def search_best_params(self):
        """ モデルの最適なパラメーターを探索 """
        # NDCGスコアは高いほど良いので direction='maximize' を指定
        study = opt.create_study(direction='maximize')
        # 100回の試行（トライアル）を実行
        study.optimize(self.objective, n_trials=100)

        # ベストパラメーター
        best_params = study.best_params
        best_params.update({
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 2, 3],
            'random_state': 42,
            'verbose': -1
        })
        print(f"ベスト NDCG@3 スコア: {study.best_value:.4f}")
        print("ベストパラメータ:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # 最適な学習データに更新
        self.params = best_params
        


def main():
    """ メイン """
    sbm = SearchBestModel()

    # データの取得
    all_df = pd.read_csv("sample/sample.csv")

    # 訓練する特徴量を指定
    features = ["frame", "avg_st", "win_rates"]

    # 学習データを用意(訓練：検証が8:2の割合でランダムに分割)
    sbm.set_study_data(all_df, features, test_size=0.2)

    # 最適な特徴量を探索(明らかに不要な特徴量を除外)
    sbm.search_best_features()

    # モデルの最適なパラメーターと最適な特徴量を繰り返し探索
    best_features = sbm.features.copy()
    best_params = sbm.params.copy()
    best_index_total = sbm.index_total
    for _ in range(10):
        # モデルの最適なパラメーターを探索
        sbm.search_best_params()
        # 再度最適な特徴量を探索(学習データや特徴量の更新は行わない)
        sbm.search_best_features(update=False)
        # 合計インデックスが小さければ、最適値更新
        if sbm.index_total <= best_index_total:
            best_params = sbm.params
            best_features = sbm.features
            best_index_total = sbm.index_total

    print(best_index_total, best_features)
    for key, value in best_params.items():
        print(f"  {key}: {value}")


main()