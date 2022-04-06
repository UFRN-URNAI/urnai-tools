import ast
import importlib
import os
import pkgutil

from .error import ClassNotFoundError


def get_class_import_path(pkg_str, classname):
    module_path = ''
    try:
        mod = importlib.import_module(pkg_str)
        for module_info in pkgutil.iter_modules(mod.__path__):
            if module_info.ispkg:
                module_path += get_class_import_path(pkg_str + '.' + module_info.name, classname)
            else:
                file_path = module_info.module_finder.path + os.path.sep + module_info.name + '.py'
                with open(file_path, 'r') as input_file:
                    node = ast.parse(input_file.read())
                    classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
                    for class_node in classes:
                        if class_node.name == classname:
                            module_path += '.'.join([pkg_str, module_info.name, classname])
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
        raise ClassNotFoundError('{} was not found in {}'.format(classname, pkg))

    cls_str = cls_str_full.pop()
    pkg_str = cls_str_full

    mod = importlib.import_module('.'.join(pkg_str))
    cls = getattr(mod, cls_str)

    return cls
