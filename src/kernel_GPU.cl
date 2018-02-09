__kernel
void mandelbrot(const double x_min, const double y_max, const double dx, const double dy, 
                const ushort max_iter, const ushort color, __global ushort *output)
{
    const ushort Idx = get_global_id(0);
    const ushort Idy = get_global_id(1);
    const int Index = Idx + get_global_size(0) * Idy;
        
    const double c_re = x_min + Idx * dx;
    const double c_im = y_max - Idy * dy;
    double re = 0;
    double im = 0;
    double re_2 = 0;
    double im_2 = 0;
    ushort i;

    for(i = 0; i < max_iter && re_2 + im_2 < 4; i++) 
        {
        im = re*im;
        im += im;
        im += c_im;
        re = re_2 - im_2 + c_re;
        re_2 = re*re;
        im_2 = im*im;
        }
    if(i == max_iter) output[Index] = i % color;
    else output[Index] = (i-1) % color;
}