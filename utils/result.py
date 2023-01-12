import pyfiglet
from pandas import option_context, DataFrame


def print_banner(text):
    f = pyfiglet.Figlet(font="slant")
    print(f.renderText(text))


def print_df(df: DataFrame):
    with option_context('expand_frame_repr', False, 'display.max_rows', None):
        print(df)
