from dataclasses import dataclass

import pandas as pd


@dataclass
class Summary:
    values: pd.DataFrame

    def add(self, values_to_add: pd.DataFrame) -> None:
        self.values = pd.concat((self.values, values_to_add))

    @property
    def model_columns(self) -> list[str]:
        return ["model_name", "lambda_", "iteration"]

    @property
    def context_constraint_suffix(self) -> str:
        return "_context_constraint"

    @property
    def known_mean_lb_suffix(self) -> str:
        return "_lower_known_constraint"

    @property
    def known_mean_ub_suffix(self) -> str:
        return "_upper_known_constraint"

    @property
    def relaxed_suffix(self) -> str:
        return "_relaxed"

    @property
    def objective(self) -> pd.Series:
        return self.values["objective"]

    @property
    def relaxed_constraints(self) -> list[str]:
        return [c for c in self.values.columns if self.relaxed_suffix in c]

    @property
    def hidden_constraints(self) -> list[str]:
        return [c for c in self.values.columns if self.context_constraint_suffix in c and self.relaxed_suffix not in c]

    @property
    def relaxed_hidden_constraints(self) -> list[str]:
        return [c for c in self.values.columns if self.context_constraint_suffix in c and self.relaxed_suffix in c]

    @property
    def known_lb_constraints(self) -> list[str]:
        return [c for c in self.values.columns if self.known_mean_lb_suffix in c]

    @property
    def known_ub_constraints(self) -> list[str]:
        return [c for c in self.values.columns if self.known_mean_ub_suffix in c]

    def check_feasibility(self, columns: list[str]) -> pd.DataFrame:
        satisfaction = self.check_satisfaction(columns)
        return satisfaction.all(axis=1)

    def check_satisfaction(self, columns: list[str]) -> pd.DataFrame:
        all_constraints = self.values[columns]
        satisfaction = all_constraints.where(all_constraints.isna(), all_constraints >= 0)
        return satisfaction.astype(float)

    def filter_by_dataset(self, validation: bool = False, test: bool = False):
        datasets_to_filter: set[str] = set()
        if validation:
            datasets_to_filter.add("validation")
        if test:
            datasets_to_filter.add("test")
        self.values = self.values[self.values.dataset.isin(datasets_to_filter)]

    def summarize_final_iterations(self) -> None:
        df = self.check_satisfaction(self.relaxed_hidden_constraints)
        df[["objective", *self.model_columns]] = self.values[["objective", *self.model_columns]]
        df["feasibility"] = self.check_feasibility(self.relaxed_constraints)

        aggregate_df = df.groupby(self.model_columns, dropna=False).mean()
        final_model_params = df[self.model_columns].groupby(["model_name", "lambda_"], as_index=False, dropna=False).max()
        final_summaries = aggregate_df.loc[final_model_params.set_index(self.model_columns).index]
        final_summaries[[*self.relaxed_hidden_constraints, "feasibility"]] *= 100
        return final_summaries.droplevel(2, axis=0).round(1).T

    def set_iteration_one_for_lambdas(self, base_generator: str = "gan_generator") -> None:
        base_generator_values = self.values[self.values.model_name == base_generator]
        lambdas = set(self.values.lambda_.unique())
        for l in lambdas:
            if l is not None:
                iteration_1_values = base_generator_values.copy()
                iteration_1_values.lambda_ = l
                self.values = pd.concat((self.values, iteration_1_values))
