#!/usr/bin/env python3

from numpy import ndarray, array, zeros

class ReadWrite:
    def __init__(self, name):
        self.header_name = "Begin_{0}".format(name, end='')
        self.footer_name = "End_{0}".format(name, end='')

    def write_header(self, fid):
        fid.write("{0}\n".format(self.header_name))

    def write_footer(self, fid):
        fid.write("{0}\n".format(self.footer_name))

    def check_header(self, fid):
        l_str = fid.readline()
        if not l_str.startswith(self.header_name):
            raise ValueError("Header incorrect on file read {0}".format(self.header_name))
        return self.get_class_member(l_str)

    def check_footer(self, l_str, b_assert=True):
        """
        :type l_str: str
        :type b_assert: bool
        """
        if not l_str.startswith(self.footer_name):
            if b_assert:
                raise ValueError("Footer incorrect on file read {0}".format(self.header_name))
            else:
                return False
        return True

    @staticmethod
    def get_vals_only(l_str, n=0):
        ret_list = []
        if n > 0:
            str_lists = l_str.split('[')
            for s in str_lists:
                vals = ReadWrite.get_vals_only(s)
                if vals:
                    ret_list.append(vals)
        else:
            vals = l_str.strip('[,]\n').split()
            for v in vals:
                v_clean = v.strip('[,]\n')
                try:
                    ret_list.append(int(v_clean))
                except ValueError:
                    try:
                        ret_list.append(float(v_clean))
                    except ValueError:
                        ret_list.append(v_clean)

        if len(ret_list) == 1:
            count_bracket = l_str.count('[')
            if count_bracket == 0:
                ret_list = ret_list[0]
        return ret_list

    def get_class_member(self, l_str):
        vals = l_str.split(maxsplit=1)
        method_name = vals[0].strip('[,]\n')
        ret_list = []
        n_read = 0

        try:
            check_type = vals[1].split()
            n_read = int(check_type[1])
            type_check = check_type[0]
            if type_check == "dict":
                ret_list = {}
            elif type_check == "ndarray":
                dim_two = int(check_type[2])
                if dim_two is 0:
                    ret_list = array([0 for _ in range(0, n_read)])
                else:
                    ret_list = zeros([n_read, dim_two])
            elif not type_check == "list":
                raise ValueError
        except (ValueError, IndexError):
            try:
                ret_list = self.get_vals_only(vals[1])
            except IndexError:
                ret_list = []

        return method_name, n_read, ret_list

    def write_class_member(self, fid, member_name, member_value):
        """
        :type fid: file
        :param fid: file name
        :param member_name: Name of class member
        :param member_value: Value of class member
        :return: None
        """
        # Dictionary
        if isinstance(member_value, dict):
            fid.write("{0} dict {1}\n".format(member_name, len(member_value)))
            for k, v in member_value.items():
                fid.write(" {0} {1}\n".format(k, v))
        elif isinstance(member_value, list):
            # list
            fid.write("{0} list {1}\n".format(member_name, len(member_value)))
            fid.write(" {0}\n".format(member_value))
        elif isinstance(member_value, ndarray):
            # ndarray
            dims = member_value.shape
            try:
                fid.write("{0} ndarray {1} {2}\n".format(member_name, dims[0], dims[1]))
            except IndexError:
                fid.write("{0} ndarray {1} 0\n".format(member_name, dims[0]))
            fid.write(" {0}\n".format(member_value))
        else:
            # Single element
            fid.write("{0} {1}\n".format(member_name, member_value))

    def write_class_members(self, fid, dir_self, class_name, exclude_list=None):
        """
        :type fid: file
        :param fid: file name
        :param dir_self: dir(self) - list of class members
        :param class_name: class name of self
        :param exclude_list: list of strings of member names to exclude
        :return: None
        """
        full_exclude_list = ["header_name", "footer_name"]
        if exclude_list:
            full_exclude_list.extend(exclude_list)

        for mem_name in dir_self:
            if mem_name in full_exclude_list:
                continue

            if hasattr(class_name, mem_name):
                continue

            atr = getattr(self, mem_name)
            self.write_class_member(fid, mem_name, atr)

    def write_check(self, fid):
        self.write_header(fid)
        self.write_class_members(fid, dir(self), ReadWrite)
        self.write_footer(fid)

    def read_class_members(self, fid, exclude_list=None):
        b_found_footer = False
        n_read = 0
        vals = []
        member_name = ""
        for l_str in fid:
            if self.check_footer(l_str, b_assert=False):
                b_found_footer = True
                break
            if n_read > 0:
                if isinstance(vals, list):
                    vals = self.get_vals_only(l_str, n_read)
                    if len(vals) != n_read:
                        setattr(self, member_name, vals[0])
                    else:
                        setattr(self, member_name, vals)
                    if n_read != len(getattr(self, member_name)):
                        raise ValueError("List size not what is written to file {0} {1}".foramt(n_read, len(vals)))
                    n_read = 0
                elif isinstance(vals, ndarray):
                    if len(vals.shape) == 1:
                        vals_row = array(self.get_vals_only(l_str, n_read))
                        vals = array(vals_row[0])
                        setattr(self, member_name, vals)
                        n_read = 0
                    else:
                        vals_row = array(self.get_vals_only(l_str, n_read))
                        vals[vals.shape[1] - n_read] = vals_row[0]
                        n_read = n_read - 1
                elif isinstance(vals, dict):
                    key_value_str = l_str.split(maxsplit=1)
                    key_val = self.get_vals_only(key_value_str[0])
                    item_val = self.get_vals_only(key_value_str[1])
                    vals[key_val] = item_val
                    n_read = n_read - 1
                    if n_read == 0:
                        setattr(self, member_name, vals)
            else:
                member_name, n_read, vals = self.get_class_member(l_str)
                if exclude_list and member_name in exclude_list:
                    return member_name, n_read, vals
                setattr(self, member_name, vals)
        if b_found_footer == False:
            raise ValueError("Did not find footer")
        return member_name, n_read, vals

    def read_check(self, fid):
        self.check_header(fid)
        self.read_class_members(fid)


if __name__ == '__main__':
    rw = ReadWrite("READWRITE")
    rw.single_int_val = 1
    rw.singe_float_val = 2.0
    rw.list_int_val = [1, 2, 3]
    rw.list_float_val = [2.0, 3.0, 4.0]
    rw.list_list = []
    rw.list_list.append([2, 3.0, 4])
    rw.list_list.append([1.0, 1, 4, 5])
    rw.dict_val = {}
    rw.dict_val[0] = "hello"
    rw.dict_val[1] = 1
    rw.dict_val[2] = [1, 2, 3]
    rw.ndarray_val = array([10, 11.2, 12])
    rw.ndarray_mat_val = array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    with open("data/RWCheck.txt", "w") as f:
        rw.write_check(f)

    rw_check = ReadWrite("READWRITE")
    with open("data/RWCheck.txt", "r") as f:
        rw_check.read_check(f)

    for d in dir(rw):
        if not hasattr(ReadWrite, d):
            if isinstance(getattr(rw, d), ndarray):
                v1 = getattr(rw, d)
                v2 = getattr(rw_check, d)
                if v1.shape != v2.shape:
                    raise ValueError("Read Write check failed, attribute {0}".format(d))

                try:
                    for i in range(0, len(v1)):
                        for j, v in enumerate(v1[i]):
                            if v != v2[i, j]:
                                raise ValueError("Read Write check failed, attribute {0}".format(d))
                except TypeError:
                    for i, v in enumerate(v1):
                        if v != v2[i]:
                            raise ValueError("Read Write check failed, attribute {0}".format(d))

            elif getattr(rw, d) != getattr(rw_check, d):
                raise ValueError("Read Write check failed, attribute {0}".format(d))
