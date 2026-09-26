import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
import optuna
import numpy as np
from itertools import permutations


def create_study_data(df, features, test_size=0.2):
    """ 学習用データを用意 """
    # 訓練データと検証データをランダムに分割（test_size=0.2の場合、グループ数の2割を検証用に指定）
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['race_id']))
    train_df:pd.DataFrame = df.iloc[train_idx]
    test_df:pd.DataFrame = df.iloc[test_idx]

    # 特徴量(X) と 正解ラベル(y)
    X_train, y_train = train_df[features], train_df["relevance"]
    X_test, y_test = test_df[features], test_df["relevance"]

    # グループ情報の作成（各レースに含まれる艇数の配列）
    group_train = train_df.groupby("race_id").size().values
    group_test = test_df.groupby("race_id").size().values

    return train_df, test_df, X_train, y_train, X_test, y_test, group_train, group_test


def create_study_model(X_train, y_train, X_test, y_test, group_train, group_test):
    """ 学習モデルを作成 """
    # モデルの型を定義
    model = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',            # ランキング学習の評価指標
        ndcg_eval_at=[1, 2, 3],    # 上位1〜3着の精度を重視
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )

    # group パラメータを渡して学習
    model.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_test, y_test)],
        eval_group=[group_test],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    return model


def get_hit_index_total(df:pd.DataFrame):
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


def main():
    """ メイン """
    # データの取得
    all_df = pd.read_csv("sample/sample.csv")

    # 訓練する特徴を指定
    features = ["frame", "avg_st", "win_rates"]

    # 学習データを用意(訓練：検証が8:2の割合でランダムに分割)
    # study_data == train_df, test_df, X_train, y_train, X_test, y_test, group_train, group_test
    study_data = create_study_data(all_df, features, test_size=0.2)

    # 学習モデル作成
    model = create_study_model(*study_data[2:8])

    # テストデータでの精度評価
    test_df = study_data[1]
    test_df["forecast"] = model.predict(test_df[features])
    # 120通りの予測から的中したインデックスの合計を取得
    index_total = get_hit_index_total(test_df)
    print(index_total)

    # 1. 特徴量重要度の取得（gain: 精度向上への貢献度）
    importance_gain = model.booster_.feature_importance(importance_type='gain')
    importance_split = model.booster_.feature_importance(importance_type='split')

    # 2. DataFrame にまとめる
    feature_imp = pd.DataFrame({
        'feature': features,
        'importance_gain (貢献度)': importance_gain,
        'importance_split (分割回数)': importance_split
    }).sort_values('importance_gain (貢献度)', ascending=False).reset_index(drop=True)

    # 3. 画面に表示
    print("\n" + "="*50)
    print("       【特徴量重要度 (Feature Importance)】")
    print("="*50)
    print(feature_imp.to_string(index=False))
    print("="*50)

main()