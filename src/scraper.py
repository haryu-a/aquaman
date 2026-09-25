""" 過去データの収集(スクレイピング) """
import pathlib
import random
import requests
from bs4 import BeautifulSoup
from itertools import permutations


class Scraper:
    file_format = "D:/DATA/aqua_man/raw/official/html/{}/{}/{}R_race_card.html"
    url_format = "https://www.boatrace.jp/owpc/pc/race/racelist?rno={}&jcd={:02}&hd={}"
    data = None

    def __init__(self, date, place, race=None):
        self.date = date
        self.place = place
        self.race = race
        html_text = self.get_html()
        if html_text:
            self.data = self.get_data(html_text)

    def get_html(self):
        # すでに取得済みのHTMLがあればそれを返す
        date, place, race = self.date, self.place, self.race
        words = [w for w in [date, place, race] if w]
        file_path = pathlib.Path(self.file_format.format(*words))
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                html_text = f.read()
                return BeautifulSoup(html_text, "html.parser")
        else:
            # HTMLがなければ、スクレイピングする
            # 代表的なブラウザのUser-Agentリスト
            ua_list = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1"
            ]
            headers = {
                'User-Agent': random.choice(ua_list)
            }
            words = [w for w in [race, place, date] if w]
            url = self.url_format.format(*words)
            response = requests.get(url, headers=headers, timeout=(5.0, 30.0))
            if response.ok:
                if "データがありません。" in response.text:
                    return None
                # 取得したHTMLをファイルに保存
                if not file_path.parent.exists():
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                return BeautifulSoup(response.text, "html.parser")
        return None

    def get_data(self, html_text: BeautifulSoup = None):
        if html_text is None:
            return {}


class ScraperRaceCard(Scraper):
    file_format = "D:/DATA/aqua_man/raw/official/html/{}/{}/{}R_race_card.html"
    url_format = "https://www.boatrace.jp/owpc/pc/race/racelist?rno={}&jcd={:02}&hd={}"

    def __init__(self, date, place, race):
        super().__init__(date, place, race)

    def get_data(self, html_text: BeautifulSoup = None):
        super().get_data(html_text)
        # 選手を取得
        racers = [row.text.replace("　", "").strip() for row in html_text.select("div.is-fs18")]
        # 等級を取得
        classes = [row.text for row in html_text.select("div.is-fs11 span")]
        # 支部を取得
        branches = [row.text.split("/")[0] for row in html_text.select("div.is-fs11")[1::2]]
        # 年齢を取得
        ages = [row.text.split()[1].split("/")[0] for row in html_text.select("div.is-fs11")[1::2]]
        # 体重を取得
        weights = [row.text.split()[1].split("/")[1] for row in html_text.select("div.is-fs11")[1::2]]
        # フライング数を取得
        f_counts = [row.text.split()[0] for row in html_text.select("td.is-lineH2")[::5]]
        # 平均スタートタイミングを取得
        avg_sts = [row.text.split()[2] for row in html_text.select("td.is-lineH2")[::5]]
        # 全国の勝率を取得
        win_rates = [row.text.split()[0] for row in html_text.select("td.is-lineH2")[1::5]]
        # 全国の2連率を取得
        top2_rates = [row.text.split()[1] for row in html_text.select("td.is-lineH2")[1::5]]
        # 全国の3連率を取得
        top3_rates = [row.text.split()[2] for row in html_text.select("td.is-lineH2")[1::5]]
        # 当地の勝率を取得
        win_rates_local = [row.text.split()[0] for row in html_text.select("td.is-lineH2")[2::5]]
        # 当地の2連率を取得
        top2_rates_local = [row.text.split()[1] for row in html_text.select("td.is-lineH2")[2::5]]
        # 当地の3連率を取得
        top3_rates_local = [row.text.split()[2] for row in html_text.select("td.is-lineH2")[2::5]]
        # モーターの2連率を取得
        top2_rates_motor = [row.text.split()[1] for row in html_text.select("td.is-lineH2")[3::5]]
        # モーターの3連率を取得
        top3_rates_motor = [row.text.split()[2] for row in html_text.select("td.is-lineH2")[3::5]]
        # ボートの2連率を取得
        top2_rates_boat = [row.text.split()[1] for row in html_text.select("td.is-lineH2")[4::5]]
        # ボートの3連率を取得
        top3_rates_boat = [row.text.split()[2] for row in html_text.select("td.is-lineH2")[4::5]]
        # 今節のスタートタイミングを取得
        series_sts = []
        series_sts.append([float(row.text) for row in html_text.select("td")[51:65] if row.text.startswith(".")])
        series_sts.append([float(row.text) for row in html_text.select("td")[117:131] if row.text.startswith(".")])
        series_sts.append([float(row.text) for row in html_text.select("td")[183:197] if row.text.startswith(".")])
        series_sts.append([float(row.text) for row in html_text.select("td")[249:263] if row.text.startswith(".")])
        series_sts.append([float(row.text) for row in html_text.select("td")[315:329] if row.text.startswith(".")])
        series_sts.append([float(row.text) for row in html_text.select("td")[381:395] if row.text.startswith(".")])
        # 今節の平均スタートタイミングを算出
        series_avg_sts = [str(round(sum(row)/len(row), 2)) for row in series_sts if len(row) > 1]
        data = {
            "racers": racers,
            "classes": classes,
            "branches": branches,
            "ages": ages,
            "weights": weights,
            "f_counts": f_counts,
            "avg_st": avg_sts,
            "win_rates": win_rates,
            "top2_rates": top2_rates,
            "top3_rates": top3_rates,
            "win_rates_local": win_rates_local,
            "top2_rates_local": top2_rates_local,
            "top3_rates_local": top3_rates_local,
            "top2_rates_motor": top2_rates_motor,
            "top3_rates_motor": top3_rates_motor,
            "top2_rates_boat": top2_rates_boat,
            "top3_rates_boat": top3_rates_boat,
            "series_avg_sts": series_avg_sts
        }
        return data


class ScraperExhibition(Scraper):
    file_format = "D:/DATA/aqua_man/raw/official/html/{}/{}/{}R_exhibition.html"
    url_format = "https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={}&jcd={:02}&hd={}"

    def __init__(self, date, place, race):
        super().__init__(date, place, race)

    def get_data(self, html_text: BeautifulSoup = None):
        super().get_data(html_text)
        # 大会のランクを取得
        race_rank = html_text.select("div.heading2_title")
        if race_rank:
            race_rank = race_rank[0].get("class")[-1]
            if "ippan" in race_rank:
                race_rank = "N"
            elif "G3" in race_rank:
                race_rank = "G3"
            elif "G2" in race_rank:
                race_rank = "G2"
            elif "G1" in race_rank:
                race_rank = "G1"
            elif "SG" in race_rank:
                race_rank = "SG"
        else:
            race_rank = None
        # 気温を取得
        temperature = html_text.select("span.weather1_bodyUnitLabelData")
        temperature = temperature[0].text.strip()[:-1] if temperature else None
        # 天気を取得
        weather = html_text.select("span.weather1_bodyUnitLabelTitle")
        weather = weather[1].text.strip() if weather else None
        # 風速を取得
        wind_speed = html_text.select("span.weather1_bodyUnitLabelData")
        wind_speed = wind_speed[1].text.strip()[:-1] if wind_speed else None
        # 風向を取得
        wind_direction = html_text.select("p.weather1_bodyUnitImage")
        if wind_direction:
            wind_direction = wind_direction[2].get("class")[-1]
            direction_num = int(wind_direction.replace("is-wind", ""))
            if direction_num in [1]:
                wind_direction = "内"
            elif direction_num in [2, 3, 4, 5, 6, 7, 8]:
                wind_direction = "追い"
            elif direction_num in [9]:
                wind_direction = "外"
            elif direction_num in [10, 11, 12, 13, 14, 15, 16]:
                wind_direction = "向かい"
            elif direction_num in [17]:
                wind_direction = "無"
        else:
            wind_direction = None
        # 水温を取得
        water_temperature = html_text.select("span.weather1_bodyUnitLabelData")
        water_temperature = water_temperature[2].text.strip()[:-1] if water_temperature else None
        # 波高を取得
        wave_height = html_text.select("span.weather1_bodyUnitLabelData")
        wave_height = wave_height[3].text.strip()[:-2] if wave_height else None
        data = {
            "race_rank": race_rank,
            "temperature": temperature,
            "weather": weather,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "water_temperature": water_temperature,
            "wave_height": wave_height
        }
        return data


class ScraperResult(Scraper):
    file_format = "D:/DATA/aqua_man/raw/official/html/{}/{}/{}R_result.html"
    url_format = "https://www.boatrace.jp/owpc/pc/race/raceresult?rno={}&jcd={:02}&hd={}"

    def __init__(self, date, place, race):
        super().__init__(date, place, race)

    def get_data(self, html_text: BeautifulSoup = None):
        super().get_data(html_text)
        # 順位を取得
        rank = [int(row.text) if row.text.isdigit() else None for row in html_text.select("td.is-fs14")[1::2]]
        # 枠番を取得
        frame = [int(row.text) for row in html_text.select("td.is-fs14")[2::2]]
        # 選手を取得
        racers = [row.text.replace("　", "").strip() for row in html_text.select("span.is-fs18")]
        # スタートタイミングを取得
        approach_route = [int(row.text) for row in html_text.select("span.table1_boatImage1Number")]
        st = [row.text.split("\n")[0].replace("　", "").strip() for row in html_text.select("span.table1_boatImage1TimeInner")]
        st = [float(s) if s[1:].isdigit() else None for s in st] # .xx → 0.xxへの数値化対応
        st = [st[i] if i < len(st) else None for i in range(6)] # 欠場対応
        st = [st[i-1] for i in approach_route] # 1-6の並びに変更
        st = [st[i-1] for i in frame] # 順位の並びに変更

        # 気温を取得
        temperature = html_text.select("span.weather1_bodyUnitLabelData")
        temperature = temperature[0].text.strip()[:-1] if temperature else None
        # 天気を取得
        weather = html_text.select("span.weather1_bodyUnitLabelTitle")
        weather = weather[1].text.strip() if weather else None
        # 風速を取得
        wind_speed = html_text.select("span.weather1_bodyUnitLabelData")
        wind_speed = wind_speed[1].text.strip()[:-1] if wind_speed else None
        # 風向を取得
        wind_direction = html_text.select("p.weather1_bodyUnitImage")
        if wind_direction:
            wind_direction = wind_direction[2].get("class")[-1]
            direction_num = int(wind_direction.replace("is-wind", ""))
            if direction_num in [1]:
                wind_direction = "内"
            elif direction_num in [2, 3, 4, 5, 6, 7, 8]:
                wind_direction = "追い"
            elif direction_num in [9]:
                wind_direction = "外"
            elif direction_num in [10, 11, 12, 13, 14, 15, 16]:
                wind_direction = "向かい"
            elif direction_num in [17]:
                wind_direction = "無"
        else:
            wind_direction = None
        # 水温を取得
        water_temperature = html_text.select("span.weather1_bodyUnitLabelData")
        water_temperature = water_temperature[2].text.strip()[:-1] if water_temperature else None
        # 波高を取得
        wave_height = html_text.select("span.weather1_bodyUnitLabelData")
        wave_height = wave_height[3].text.strip()[:-2] if wave_height else None
        data = {
            "rank": rank,
            "frame": frame,
            "racers": racers,
            "st": st,
            "temperature": temperature,
            "weather": weather,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "water_temperature": water_temperature,
            "wave_height": wave_height
        }
        return data


class ScraperOdds(Scraper):
    file_format = "D:/DATA/aqua_man/raw/official/html/{}/{}/{}R_odds.html"
    url_format = "https://www.boatrace.jp/owpc/pc/race/odds3t?rno={}&jcd={:02}&hd={}"

    def __init__(self, date, place, race):
        super().__init__(date, place, race)

    def get_data(self, html_text: BeautifulSoup = None):
        super().get_data(html_text)
        # オッズ取得
        odds_list = []
        odds_list += [row.text for row in html_text.select("td.oddsPoint")[0::6]] # 1号艇1着のオッズ
        odds_list += [row.text for row in html_text.select("td.oddsPoint")[1::6]] # 2号艇1着のオッズ
        odds_list += [row.text for row in html_text.select("td.oddsPoint")[2::6]] # 3号艇1着のオッズ
        odds_list += [row.text for row in html_text.select("td.oddsPoint")[3::6]] # 4号艇1着のオッズ
        odds_list += [row.text for row in html_text.select("td.oddsPoint")[4::6]] # 5号艇1着のオッズ
        odds_list += [row.text for row in html_text.select("td.oddsPoint")[5::6]] # 6号艇1着のオッズ
    
        # 買い目リスト作成
        order_list = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]
        
        # オッズ情報作成
        data = {order : float(odds) if odds != "欠場" else 0 for order, odds in zip(order_list, odds_list)}
        return data


class ScraperRaceIndex(Scraper):
    file_format = "D:/DATA/aqua_man/raw/official/html/{}/{}/raceindex.html"
    url_format = "https://www.boatrace.jp/owpc/pc/race/raceindex?jcd={:02}&hd={}"

    def __init__(self, date, place):
        super().__init__(date, place)

    def get_data(self, html_text: BeautifulSoup = None):
        super().get_data(html_text)
        data = []
        for tbody in html_text.find_all("tbody"):
            race, time = [row.text.strip() for row in tbody.select("td")[:2]]
            data.append([race, time])
        return data


def get_all_race_index(date):
    """ 全レースの日程を取得 """
    all_race_index = []
    for place in range(1, 25):
        race_index = ScraperRaceIndex(date, str(place)).data
        all_race_index.extend(race_index) if race_index else None
    print(all_race_index)


if __name__ == "__main__":
    print(ScraperRaceCard(20251101, 1, 1).data)
    #get_all_race_index("20251101")