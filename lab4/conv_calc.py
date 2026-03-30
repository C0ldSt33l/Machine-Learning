def comp_cut(cut: list[list[str]], kernel: list[list[str]]) -> int:
    if len(cut) != len(kernel):
        raise Exception('Row count different')
    if len(cut[0]) != len(kernel[0]):
        raise Exception('Col count different')

    coincidences: int = 0

    for r_idx in range(len(cut)):
        for c_idx in range(len(cut)):
            cut_el = cut[r_idx][c_idx]
            kernel_el = kernel[r_idx][c_idx]

            if cut_el == kernel_el and cut_el == '1':
                coincidences += 1
    
    return coincidences


data = [
    list('001000'),
    list('010100'),
    list('100010'),
    list('010100'),
    list('001010'),
    list('000001'),
]

kernel = [
    list('001'),
    list('010'),
    list('100'),
]
kernel_size = len(kernel)

for x_offset in range(4):
    for y_offset in range(4):
        cut = [line[x_offset:x_offset+kernel_size] for line in data[y_offset:y_offset+kernel_size]]
        print(comp_cut(cut, kernel),end='|')
    print()