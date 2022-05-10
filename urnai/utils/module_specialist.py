import ast
import importlib
import inspect
import os
import pkgutil

from .error import ClassNotFoundError, EnvironmentNotSupportedError


def get_class_import_path(pkg_str, classname):
    module_path = ''
    try:
        mod = importlib.import_module(pkg_str)
        for module_info in pkgutil.iter_modules(mod.__path__):
            if module_info.ispkg:
                module_path += get_class_import_path(
                    pkg_str + '.' + module_info.name, classname,
                )
            else:
                file_path = (
                    module_info.module_finder.path
                    + os.path.sep
                    + module_info.name
                    + '.py'
                )
                with open(file_path, 'r') as input_file:
                    node = ast.parse(input_file.read())
                    classes = [
                        n for n in node.body if isinstance(
                            n, ast.ClassDef)]
                    for class_node in classes:
                        if class_node.name == classname:
                            module_path += '.'.join(
                                [pkg_str, module_info.name, classname],
                            )
    except ModuleNotFoundError:
        pass
    except NameError:
        pass

    return module_path


def get_cls(pkg, classname):
    cls_str_full = get_class_import_path(pkg, classname)
    if cls_str_full != '':
        cls_str_full = cls_str_full.split('.')
    else:
        raise ClassNotFoundError(
            '{} was not found in {}'.format(
                classname, pkg))

    cls_str = cls_str_full.pop()
    pkg_str = cls_str_full

    mod = importlib.import_module('.'.join(pkg_str))
    cls = getattr(mod, cls_str)

    return cls


def get_classes_recursively(pkg_path, ignore=[]):
    cls_list = []

    try:
        mod = importlib.import_module(pkg_path)
        for module_info in pkgutil.iter_modules(mod.__path__):
            if module_info.ispkg:
                cls_list += get_classes_recursively(
                    pkg_path + '.' + module_info.name, ignore,
                )
            else:
                file_path = (
                    module_info.module_finder.path
                    + os.path.sep
                    + module_info.name
                    + '.py'
                )
                with open(file_path, 'r') as input_file:
                    node = ast.parse(input_file.read())
                    cls_list += [
                        n.name for n in node.body if isinstance(
                            n, ast.ClassDef)]

    except ModuleNotFoundError:
        pass
    except NameError:
        pass

    for cls_name in ignore:
        try:
            cls_list.remove(cls_name)
        except ValueError:
            continue

    return cls_list


def get_class_parameters(pkg_path, classname):
    try:
        cls = get_cls(pkg_path, classname)
    except ModuleNotFoundError:
        if pkg_path == 'urnai.envs':
            raise EnvironmentNotSupportedError(
                '{} returned a ModuleNotFoundError, is {} installed correctly?'.format(
                    classname, classname))
        else:
            raise

    if cls is not None:
        params = inspect.getargspec(cls.__init__)
        names = params.args
        defaults = params.defaults

        default_names = names[(len(names) - len(defaults)):]
        names = names[0: (len(names) - len(defaults))]
        names.pop(0)

        param_data = {
            'params_without_defaults': names,
            'params_with_deaults': []}

        for param, default_value in zip(default_names, defaults):
            param_data['params_with_deaults'].append(
                {'param': param, 'default_value': default_value},
            )

        return param_data
