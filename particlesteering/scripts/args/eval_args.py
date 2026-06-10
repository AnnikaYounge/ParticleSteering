# Adapted from AxBench — https://github.com/stanfordnlp/axbench (Apache-2.0).
from dataclasses import dataclass, field
import argparse
import yaml
from typing import Optional, List, Type, get_args, get_origin

@dataclass
class EvalArgs:
    # Steering vs latent vs both (`evaluate.py --mode`; YAML key matches argparse).
    mode: Optional[str] = "all"
    models: List[str] = field(default_factory=lambda: [])
    steering_evaluators: List[str] = field(default_factory=lambda: [])
    report_to: List[str] = field(default_factory=lambda: [])
    data_dir: Optional[str] = None
    dump_dir: Optional[str] = None
    num_of_workers: Optional[int] = 16
    lm_model: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    master_data_dir: Optional[str] = None
    overwrite_inference_dump_dir: Optional[str] = None
    overwrite_evaluate_dump_dir: Optional[str] = None
    steer_data_type: Optional[str] = "concept"
    openai_timeout: Optional[float] = 120.0
    judge_batch_size: Optional[int] = 8

    def __init__(
        self,
        description: str = "Evaluation Script",
        config_file: str = None,
        section: str = "train",  # Specify section to load
        custom_args: Optional[List[dict]] = None,
        override_config: bool = True,
        ignore_unknown: bool = False
    ):
        """
        Initializes EvalArgs by parsing command-line arguments and loading configurations from a YAML file.
        """
        parser = argparse.ArgumentParser(description=description)

        # Add config file argument
        parser.add_argument(
            '--config',
            type=str,
            default=config_file,
            help='Path to the YAML configuration file.'
        )

        # Add arguments corresponding to the dataclass fields
        fields = self.__dataclass_fields__
        for field_name, field_def in fields.items():
            if field_name == 'config_file':
                continue

            # Handle list-type fields specially for command line input
            if hasattr(field_def.type, '__origin__') and field_def.type.__origin__ is list:
                parser.add_argument(
                    f'--{field_name}',
                    nargs='+',  # This allows multiple values
                    help=f'Specify {field_name} (can provide multiple values).',
                )
            else:
                arg_type = self._get_argparse_type(field_def.type)
                parser.add_argument(
                    f'--{field_name}',
                    type=arg_type,
                    help=f'Specify {field_name}.',
                )

        # Add any custom arguments provided (skip if already registered on EvalArgs, e.g. --mode)
        registered_options = set()
        for action in parser._actions:
            registered_options.update(action.option_strings)
        if custom_args:
            for arg in custom_args:
                option_strings = tuple(arg.get('args', ()))
                if any(opt in registered_options for opt in option_strings):
                    continue
                parser.add_argument(*option_strings, **arg.get('kwargs', {}))
                registered_options.update(option_strings)

        # Use parse_known_args instead of parse_args if ignore_unknown is True
        if ignore_unknown:
            args, unknown = parser.parse_known_args()
            if unknown:
                print(f"EvalArgs: ignoring unknown arguments: {unknown}")
        else:
            args = parser.parse_args()

        # Load the YAML configuration file
        config_file_path = args.config
        if not config_file_path:
            raise ValueError("A config file must be provided.")
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)

        # Select the specified section
        section_data = config.get(section, {})
        if not section_data:
            raise ValueError(f"Section '{section}' not found in the YAML configuration.")

        # Initialize attributes from the selected section
        for field_name in fields:
            if field_name == 'config_file':
                continue
            value = section_data.get(field_name, None)
            setattr(self, field_name, value)

        # Overwrite with command-line arguments if provided
        if override_config:
            for field_name in vars(args):
                if field_name in ['config']:
                    continue
                arg_value = getattr(args, field_name)
                if arg_value is not None:
                    setattr(self, field_name, arg_value)

        # CLI int fields may arrive as strings (e.g. Optional[int] typing on py3.12).
        if self.num_of_workers is not None:
            self.num_of_workers = int(self.num_of_workers)
        if self.judge_batch_size is not None:
            self.judge_batch_size = int(self.judge_batch_size)
        if self.openai_timeout is not None:
            self.openai_timeout = float(self.openai_timeout)

        # Additional attributes
        self.config_file = config_file_path

    @staticmethod
    def _get_argparse_type(field_type: Type) -> Type:
        """
        Helper method to get the argparse type from the dataclass field type.
        """
        # Optional[T] is Union[T, None] on Python 3.10+; unwrap to T for argparse.
        if get_origin(field_type) is not None:
            non_none = [a for a in get_args(field_type) if a is not type(None)]
            if len(non_none) == 1:
                field_type = non_none[0]
        if field_type == int:
            return int
        elif field_type == float:
            return float
        elif field_type == bool:
            return lambda x: (str(x).lower() in ['true', '1', 'yes'])
        else:
            return str