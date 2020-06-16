from pathlib import Path

import pandas as pd
from sklearn.datasets import load_boston
import openpyxl as px

from nclick.utils.log import Log
from nclick.eda.summarize import summary_to_excel


logger_name = 'log_check_data'
output_dir = Path('./result_check_data')


if __name__ == '__main__':

    # ログファイル作成
    logfile_path =  output_dir/f'{logger_name}.log'
    Log.create_logger(logger_name, logfile_path)
    Log.emphasis('Make Summary', logger_name, 'info')

    # データ取得
    with Log.timer('Read Data', logger_name, 'info'):
        boston = load_boston()
        data = pd.DataFrame(boston.data, columns=boston.feature_names)

    # 基本統計量算出
    with Log.timer('Summary Data', logger_name, 'info'):
        wb = px.Workbook()
        ws = wb.create_sheet(title='basic aggregation')
        summary_to_excel(ws, data)

    # 書き出し
    with Log.timer('Write Data', logger_name, 'info'):
        default_sheet = wb['Sheet']
        wb.remove(default_sheet)
        wb.save(output_dir/'basic_aggregation_summary.xlsx')

