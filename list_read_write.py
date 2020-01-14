#!/usr/bin/env python3


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
    def get_vals_only(l_str):
        l_str = l_str.strip('[,]\n')
        vals = l_str.split()
        ret_list = []
        for v in vals:
            v_clean = v.strip('[,]\n')
            if '.' in v_clean:
                ret_list.append(float(v_clean))
            else:
                ret_list.append(int(v_clean))
        return ret_list

    def get_class_member(self, l_str):
        vals = l_str.split(maxsplit=1)
        method_name = vals[0].strip('[,]\n')
        # Just in case empty vals list
        ret_list = []
        if len(vals) > 1:
            ret_list = self.get_vals_only(vals[1])

        return method_name, ret_list

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

            if not hasattr(class_name, mem_name):
                fid.write("{0} {1}\n".format(mem_name, getattr(self, mem_name)))

    def write_check(self, fid):
        self.write_header(fid)
        self.write_class_members(fid, dir(self), ReadWrite)
        self.write_footer(fid)

    def read_class_members(self, fid):
        b_found_footer = False
        for l_str in fid:
            if self.check_footer(l_str, b_assert=False):
                b_found_footer = True
                break
            method_name, vals = self.get_class_member(l_str)
            if len(vals) == 1:
                setattr(self, method_name, vals[0])
            else:
                setattr(self, method_name, vals)
        if b_found_footer == False:
            raise ValueError("Did not find footer")

    def read_check(self, fid):
        self.check_header(fid)
        self.read_class_members(fid)


if __name__ == '__main__':
    rw = ReadWrite("READWRITE")
    rw.single_int_val = 1
    rw.singe_float_val = 2.0
    rw.list_int_val = [1, 2, 3]
    rw.list_float_val = [2.0, 3.0, 4.0]

    with open("data/RWCheck.txt", "w") as f:
        rw.write_check(f)

    rw_check = ReadWrite("READWRITE")
    with open("data/RWCheck.txt", "r") as f:
        rw_check.read_check(f)

    for d in dir(rw):
        if not hasattr(ReadWrite, d):
            if getattr(rw, d) != getattr(rw_check, d):
                raise ValueError("Read Write check failed, attribute {0}".format(d))
