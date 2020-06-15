import numpy as np
import pandas as pd

from nclick.excel import writing_tools as wt
from nclick.excel import cell_styles as cs


def describe_more(data):
    """
    pandas DataFrameのdescribeメソッドの詳細版
    Parameters
    ----------
    data : pandas DataFrame
        describeメソッドを適用したいデータフレーム
    """
    n_rows, n_cols = data.shape
    des = data.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95],
                        include='all')
    rep = des.index.tolist()

    # NA数, 空文字を算出
    des.loc['na_num', :] = data.isna().sum(axis=0)
    des.loc['na_num_p', :] = des.loc['na_num', :]/n_rows
    des.loc['empty_num', :] = (data == '').sum(axis=0)
    rep.insert(1, 'na_num')
    rep.insert(2, 'na_num_p')
    rep.insert(3, 'empty_num')

    # 最頻値算出
    if 'freq' in rep:
        # 第一最頻値を加工
        des.rename(index={'freq': 'top_freq'}, inplace=True)
        des.loc['top_freq_p'] = des.loc['top_freq', :] / n_rows
        rep[rep.index('freq')] = 'top_freq'
        rep.insert(rep.index('top_freq') + 1, 'top_freq_p')

        # 第二、第三最頻値を算出
        modes = [[np.nan]*n_cols for i in range(6)]
        for i, col in enumerate(des.columns):
            if np.isnan(des.loc['top_freq', col]) \
               or ('first' in rep and not des.loc['first', col] is np.nan):
                continue

            vc = data[col].value_counts()
            if len(vc) > 1:
                modes[0][i] = vc.index[1]
                modes[1][i] = vc[1]
                modes[2][i] = vc[1]/n_rows
            if len(vc) > 2:
                modes[3][i] = vc.index[2]
                modes[4][i] = vc[2]
                modes[5][i] = vc[2]/n_rows

        new_cols = [
            'second', 'second_freq', 'second_freq_p',
            'third', 'third_freq', 'third_freq_p'
        ]

        insert_point = rep.index('top_freq_p') + 1
        for i, col in enumerate(new_cols):
            rep.insert(insert_point+i, col)
            des.loc[col, :] = modes[i]

    # 整列
    des = des.loc[rep, :]

    en2ja = {
        'count': '非NA数',
        'na_num': 'NA数',
        'na_num_p': 'NA割合',
        'empty_num': '空文字数',
        'unique': 'ユニーク数',
        'top': '最頻値',
        'top_freq': '最頻値数',
        'top_freq_p': '最頻値割合',
        'second': '第二最頻値数',
        'second_freq': '第二最頻値数',
        'second_freq_p': '第二最頻値割合',
        'third': '第三最頻値',
        'third_freq': '第三最頻値数',
        'third_freq_p': '第三最頻値割合',
        'first': '最古日付',
        'last': '最新日付',
        'mean': '平均値',
        'std': '標準偏差',
        'min': '最小値',
        '5%': '5%点',
        '25%': '25%点',
        '50%': '50%点',
        '75%': '75%点',
        '95%': '95%点',
        'max': '最大値',
    }

    des.index = map(lambda en: en2ja[en], des.index)
    return des


def summary_to_excel(ws,
                     data):
    """
    サマリを計算し、Excelファイルに書き込む
    Parameters
    ----------
    ws : openpyxl Worksheet
        出力対象のExcelシート
    data : pandas DataFrame
        サマライズするデータ
    """

    # サンプルのレコードを保持
    if data.shape[0] > 300:
        sample = pd.concat([data.head(150), data.tail(150)]).astype(str)
        sample.reset_index(inplace=True)
    else:
        sample = data.astype(str)
        sample.reset_index(inplace=True)

    # サマリを算出
    des = describe_more(data)
    n_rows, n_cols = des.shape
    des.reset_index(inplace=True)
    des.rename(columns={'index': ''}, inplace=True)

    # サマリの書き出し
    wt.dataframe_to_sheet(des,
                          ws,
                          start_row=1,
                          start_col=1,
                          header=True,
                          index_cols_num=1,
                          header_style=cs.style_00,
                          index_style=cs.style_00,
                          cell_style=cs.style_02)

    # サマリ乗せるにスタイルを適用
    wt.fill_cell(ws,
                 2,
                 2+n_rows,
                 2,
                 2+n_cols,
                 cs.style_01,
                 )

    # 行数、列数の書き出し
    wt.set_value(ws, '行数', n_rows+3, 1)
    wt.set_style(ws, cs.style_00, n_rows+3, 1)
    wt.set_value(ws, data.shape[0], n_rows+3, 2)
    wt.set_style(ws, cs.style_01, n_rows+3, 2)
    wt.set_value(ws, '列数', n_rows+3, 3)
    wt.set_style(ws, cs.style_00, n_rows+3, 3)
    wt.set_value(ws, data.shape[1], n_rows+3, 4)
    wt.set_style(ws, cs.style_01, n_rows+3, 4)

    # サンプルデータの書き出し
    wt.dataframe_to_sheet(sample,
                          ws,
                          start_row=n_rows + 4,
                          start_col=1,
                          header=True,
                          index_cols_num=1,
                          header_style=cs.style_00,
                          index_style=cs.style_01,
                          cell_style=cs.style_02)

    # 見栄えを整える
    wt.freeze_header_and_index(ws, 5+n_rows, 2)
    wt.autoresize_columns_width(ws, ratio=1.5)