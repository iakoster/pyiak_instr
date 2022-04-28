import shutil
import unittest
import configparser
from pathlib import Path

from pyinstr_iakoster.rwfile import RWConfig
from pyinstr_iakoster.exceptions import FilepathPatternError

from tests.env_vars import DATA_TEST_DIR


CONFIG_NAME = 'test_config.ini'
CONFIG_PATH = DATA_TEST_DIR / CONFIG_NAME
TEST_DICT_STR = {
    'test_str': {
        'test_letter': 'a',
        'test_word': 'abcde',
    },
    'test_single': {
        'test_int': '1',
        'test_float': '5.4321',
        'test_efloat_small': '5.4321e-99',
        'test_efloat_huge': '5.4321e+99',
        'test_bool': 'True',
        'test_none': 'None'
    },
    'test_list': {
        'test_str': 'a,b,c,d,e',
        'test_int': '1,2,3,4,5',
        'test_float': '1.1,2.2,3.3,4.4,5.5',
        'test_efloat_small':
            '1.1e-99,2.2e-99,3.3e-99,4.4e-99,5.5e-99',
        'test_efloat_huge':
            '1.1e+99,2.2e+99,3.3e+99,4.4e+99,5.5e+99'
    },
    'test_tuple': {
        'test_str': 'a;b;c;d;e',
        'test_int': '1;2;3;4;5'
    },
    'test_dict': {
        'test_str_str': 'a;z,b;y,c;x,d;w,e;v',
        'test_str_int': 'a;1,b;2,c;3,d;4,e;5',
        'test_float_str': '1.1;z,2.2;y,3.3;x,4.4;w,5.5;v',
        'test_int_efloat':
            '1;1.1e+45,2;2.2e+99,3;3.3e-99,40;4.4,576;5.5'
    }
}
TEST_DICT_CONV = {
    'test_str': {
        'test_letter': 'a',
        'test_word': 'abcde',
    },
    'test_single': {
        'test_int': 1,
        'test_float': 5.4321,
        'test_efloat_small': 5.4321e-99,
        'test_efloat_huge': 5.4321e+99,
        'test_bool': True,
        'test_none': None
    },
    'test_list': {
        'test_str': ['a', 'b', 'c', 'd', 'e'],
        'test_int': [1, 2, 3, 4, 5],
        'test_float': [1.1, 2.2, 3.3, 4.4, 5.5],
        'test_efloat_small':
            [1.1e-99, 2.2e-99, 3.3e-99, 4.4e-99, 5.5e-99],
        'test_efloat_huge':
            [1.1e+99, 2.2e+99, 3.3e+99, 4.4e+99, 5.5e+99]
    },
    'test_tuple': {
        'test_str': ('a', 'b', 'c', 'd', 'e'),
        'test_int': (1, 2, 3, 4, 5)
    },
    'test_dict': {
        'test_str_str': {
            'a': 'z', 'b': 'y', 'c': 'x', 'd': 'w', 'e': 'v'},
        'test_str_int': {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5},
        'test_float_str': {
            1.1: 'z', 2.2: 'y', 3.3: 'x', 4.4: 'w', 5.5: 'v'},
        'test_int_efloat': {
            1: 1.1e+45, 2: 2.2e+99, 3: 3.3e-99, 40: 4.4, 576: 5.5}
    }
}


class TestRWConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_config = RWConfig(CONFIG_PATH)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(DATA_TEST_DIR)

    def test_path_is_dir(self):
        test_path = Path('.\\data_test\\only_dir')
        with self.assertRaises(FilepathPatternError) as err:
            RWConfig(test_path)
        self.assertEqual(
            'The path does not lead to \'\\\\S+.ini$\' file',
            err.exception.message
        )

    def test_wrong_fileformat(self):
        test_path = Path('.\\date_test\\not_ini.txt')
        with self.assertRaises(FilepathPatternError) as err:
            RWConfig(test_path)
        self.assertEqual(
            'The path does not lead to \'\\\\S+.ini$\' file',
            err.exception.message
        )

    def test_file_creation(self):
        print(RWConfig(CONFIG_PATH).__class__.LIST_PATTERN.__doc__)
        RWConfig(CONFIG_PATH)
        self.assertTrue(CONFIG_PATH.exists())

    def test_filepath_as_str(self):
        RWConfig(str(CONFIG_PATH))
        self.assertTrue(CONFIG_PATH.exists())

    def test_write_single(self):
        options = ['test_int', 'test_float',
                   'test_efloat_small', 'test_efloat_huge']
        vals = [10, 254.641685, 1.12476e-123, 5.4165e+123]
        vals_str = ['10', '254.641685', '1.12476e-123', '5.4165e+123']
        for i in range(len(options)):
            with self.subTest(option=options[i]):
                self.test_config.write(
                    'test_single', options[i], vals[i])
                readed_config = configparser.ConfigParser()
                readed_config.read(CONFIG_PATH)
                self.assertEqual(
                    vals_str[i],
                    readed_config.get('test_single', options[i])
                )

    def test_write_dict_conv(self):
        self.test_config.write(TEST_DICT_CONV)
        readed_config = configparser.ConfigParser()
        readed_config.read(CONFIG_PATH)
        self.assertDictEqual(
            TEST_DICT_STR,
            {s: dict(readed_config.items(s))
             for s in readed_config.sections()}
        )

    def test_write_dict_str(self):
        self.test_config.write(TEST_DICT_STR)
        readed_config = configparser.ConfigParser()
        readed_config.read(CONFIG_PATH)
        self.assertDictEqual(
            TEST_DICT_STR,
            {s: dict(readed_config.items(s))
             for s in readed_config.sections()}
        )

    def test_write_dict_part(self):
        test_dict = {'test_single': {
            'test_int': 2,
            'test_float': 5.4321,
            'test_efloat_small': 5.4321e-99,
            'test_efloat_huge': 5.4321e+99,
        }}

        self.test_config.write(test_dict)
        readed_config = configparser.ConfigParser()
        readed_config.read(CONFIG_PATH)
        red_dict = TEST_DICT_STR
        red_dict['test_single']['test_int'] = '2'
        self.assertDictEqual(
            red_dict,
            {s: dict(readed_config.items(s))
             for s in readed_config.sections()}
        )

    def test_write_wrong_args(self):
        with self.assertRaises(TypeError) as exc:
            self.test_config.write('', '')
        self.assertEqual(
            exc.exception.args[0],
            'Wrong args for write method'
        )

    def setUp(self) -> None:
        self.test_config.write(TEST_DICT_STR)

    def test_read_conversion(self) -> None:
        high_items = TEST_DICT_CONV.items()
        for section, option_dict in high_items:
            for option, conv_value in option_dict.items():
                with self.subTest(section=section, option=option):
                    self.assertIsInstance(
                        self.test_config.read(section, option),
                        type(conv_value)
                    )
                    self.assertEqual(
                        self.test_config.read(section, option),
                        conv_value
                    )

    def test_get(self) -> None:
        section = 'test_single'
        option = 'test_int'
        self.assertEqual(
            TEST_DICT_CONV[section][option],
            self.test_config.get(section, option)
        )

    def test_set(self) -> None:
        section = 'test_single'
        option = 'test_int'
        value = 12345
        self.test_config.set(section, option, value)
        self.assertEqual(
            value,
            self.test_config.get(section, option)
        )

    def test_set_isolation(self) -> None:

        section = 'test_single'
        option = 'test_int'
        value = 12345
        self.test_config.set(section, option, value)
        self.assertEqual(
            value,
            self.test_config.get(section, option)
        )
        self.assertEqual(
            TEST_DICT_CONV[section][option],
            self.test_config.read(section, option)
        )

    def test_apply_changes(self) -> None:

        section = 'test_single'
        option = 'test_int'
        value = 12345
        self.test_config.set(section, option, value)
        self.test_config.apply_changes()
        self.assertEqual(
            value,
            self.test_config.read(section, option)
        )
        self.assertEqual(
            value,
            self.test_config.get(section, option)
        )
