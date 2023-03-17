from typing import List

import pandas as pd
from torch.utils.benchmark.utils import common
from torch.utils.benchmark.utils.compare import Compare as _Comp
from torch.utils.benchmark.utils.compare import Table


class Table_CSV(Table):
    def render(self) -> str:

        has_warnings = (
            self._highlight_warnings and any(
                ri.has_warnings
                for ri in self.results
            )
        )

        string_rows = [['arguments'] + self.column_keys + [
            'op',
            'time_unit',
            'has_warnings',
            'threads',
        ]]
        nt = None
        for r in self.rows:
            if r._num_threads is not None:
                nt = r._num_threads
            string_rows.append(r.as_column_strings())
            string_rows[-1] += [
                self.label, self.time_unit, str(has_warnings),
                str(nt),
            ]

        finalized_columns = [';'.join(r).strip() for r in string_rows]

        newline = '\n'
        return newline.join(finalized_columns)


class Compare(_Comp):
    def _layout(self, results: List[common.Measurement]):
        table = Table_CSV(
            results,
            self._colorize,
            self._trim_significant_figures,
            self._highlight_warnings,
        )
        return table.render()

    def join_to_df(self):
        out = self._render()
        headers = [o.split('\n')[0] for o in out]

        # Check if all headers are equal
        if headers.count(headers[0]) == len(headers):
            header = headers[0]
            content = [o.split('\n')[1:] for o in out]
            content = [
                item.split(';') for sublist in content
                for item in sublist
            ]

            return pd.DataFrame(
                columns=header.split(';'),
                data=content,
            )
        else:
            raise Exception(
                'Unsupported operation since the headers does'
                ' not match.',
            )
