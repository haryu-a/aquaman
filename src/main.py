import scraper as sp
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
import optuna


# --------------------------------------------------
# 1. データの準備
# --------------------------------------------------
all_df = pd.DataFrame()
races = [[20251101, 1, i] for i in range(1, 11)]
for race in races:
    # データ取得
    race_card_data = sp.ScraperRaceCard(*race).data
    result_data = sp.ScraperResult(*race).data
    # データ成形
    data = {
        "race_id": "{}{:02}{:02}".format(*race),
        "frame": list(range(1, 7)),
        "avg_st": [float(i) for i in race_card_data["avg_st"]],
        "win_rates": [float(i) for i in race_card_data["win_rates"]],
        "rank": [float(i) for i in result_data["frame"]]
    }
    # データフレーム化
    df = pd.DataFrame(data)
    #ランキング学習では「数字が大きいほど優秀」と判断されるため、実際の着順（1〜6着）を反転させたスコアを割り当て
    df['relevance'] = 6 - df['rank']  # 1着=5点 〜 6着=0点
    # rank行削除
    #df = df.drop(columns=["rank"])
    # データ統合
    all_df = pd.concat([all_df, df], ignore_index=True)

# GroupShuffleSplitの初期化（test_size=0.2 でグループ数の2割を検証用に指定）
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# 訓練データと検証データを8:2の割合でランダムに分割
train_idx, test_idx = next(gss.split(all_df, groups=all_df['race_id']))
train_df:pd.DataFrame = all_df.iloc[train_idx]
test_df:pd.DataFrame = all_df.iloc[test_idx]

features = ["frame", "avg_st", "win_rates"]
X_train, y_train = train_df[features], train_df["relevance"]
X_test, y_test = test_df[features], test_df["relevance"]

group_train = train_df.groupby("race_id").size().values
group_test = test_df.groupby("race_id").size().values

# --------------------------------------------------
# 2. Optunaの目的関数 (objective) の定義
# --------------------------------------------------
def objective(trial):
    # パラメータの探索範囲を設定
    params = {
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
    model = lgb.LGBMRanker(**params)
    
    model.fit(
        X_train, y_train,
        group=group_train,
        eval_set=[(X_test, y_test)],
        eval_group=[group_test],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    
    # 検証データでの評価スコア（ベストなNDCG@3スコア）を目的関数として返す
    # best_score_ から eval_set(0) の ndcg@3 を取得
    best_score = model.best_score_['valid_0']['ndcg@3']
    
    return best_score

# --------------------------------------------------
# 3. 最適化の実行
# --------------------------------------------------
# NDCGスコアは高いほど良いので direction='maximize' を指定
study = optuna.create_study(direction='maximize')

# 50回の試行（トライアル）を実行
study.optimize(objective, n_trials=50)

# --------------------------------------------------
# 4. 最適化結果の確認
# --------------------------------------------------
print("=== 最適化結果 ===")
print(f"ベスト NDCG@3 スコア: {study.best_value:.4f}")
print("ベストパラメータ:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# --------------------------------------------------
# 5. ベストパラメータを使って最終モデルを再学習
# --------------------------------------------------
best_params = study.best_params
best_params.update({
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1, 2, 3],
    'random_state': 42,
    'verbose': -1
})

best_model = lgb.LGBMRanker(**best_params)
best_model.fit(
    X_train, y_train,
    group=group_train,
    eval_set=[(X_test, y_test)],
    eval_group=[group_test],
    callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
)

# --------------------------------------------------
# 6. テストデータでの精度評価と print 表示
# --------------------------------------------------
# テストデータに対して予測スコアを計算してデータフレームに追加
test_df = test_df.copy() # または学習に使っていない完全未見のテストデータ
test_df['pred_score'] = best_model.predict(test_df[features])

def print_test_accuracy(df:pd.DataFrame):
    total_races = df['race_id'].nunique()
    
    win_hits = 0        # AIの1位予測が実際に1着になった数
    exact_top3_hits = 0 # AIのTOP3予測が実際の1〜3着（着順問わず）を完全網羅した数
    top3_matches = 0    # AIのTOP3予測のうち、実際に3着以内に入った艇の延べ数

    for race_id, group in df.groupby('race_id'):
        # AIの予測スコアが高い順にソート
        pred_sorted = group.sort_values('pred_score', ascending=False)
        
        # AIが予測した1着〜3着の艇（枠番）
        pred_1st_frame = pred_sorted.iloc[0]['frame']
        pred_top3_frames = set(pred_sorted.iloc[:3]['frame'])
        
        # 実際の着順結果
        actual_1st_frame = group[group['rank'] == 1]['frame'].values[0]
        actual_top3_frames = set(group[group['rank'] <= 3]['frame'])
        
        # 1. 単勝（1着）的中判定
        if pred_1st_frame == actual_1st_frame:
            win_hits += 1
            
        # 2. TOP3カバー数のカウント
        matched_count = len(pred_top3_frames.intersection(actual_top3_frames))
        top3_matches += matched_count
        
        # 3. 3連複的中（上位3艇を順番無視で全員的中）判定
        if matched_count == 3:
            exact_top3_hits += 1

    # 各種精度の計算
    win_rate = (win_hits / total_races) * 100
    top3_trio_rate = (exact_top3_hits / total_races) * 100
    top3_precision = (top3_matches / (total_races * 3)) * 100

    # 画面表示
    print("\n" + "="*45)
    print("        【テストデータ（未知レース）精度評価】")
    print("="*45)
    print(f" 評価レース数       : {total_races:,} レース")
    print(f" 1着的中率 (単勝)   : {win_rate:.2f}%  ({win_hits}/{total_races})")
    print(f" 3連複ボックス的中率: {top3_trio_rate:.2f}%  ({exact_top3_hits}/{total_races})")
    print(f" TOP3精度 (適合率)  : {top3_precision:.2f}%  (予測上位3艇のうち実際に3着以内に入った割合)")
    print("="*45)

# 実行
print_test_accuracy(test_df)

