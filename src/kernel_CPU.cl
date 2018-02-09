__kernel
__attribute__((vec_type_hint(double8))) 
void mandelbrot(const double x_min, const double y_max, const double dx, const double dy, 
                const ushort max_iter, const ushort color, __global ushort *output)
{
    const ushort Idx = get_global_id(0) * 8;
    const ushort Idy = get_global_id(1);
    const int Index = Idx + 8 * get_global_size(0) * Idy;

    const double8 vdx = (double8)(dx);
    const double8 vdy = (double8)(dy);
    const double8 vx_min = (double8)(x_min);
    const double8 vy_max = (double8)(y_max);
    const double8 fours = (double8)(4);
    const double8 vIdx = (double8)(Idx) + (double8)(0,1,2,3,4,5,6,7);
    const double8 vIdy = (double8)(Idy);

    const double8 c_re = vx_min + vIdx * vdx;
    const double8 c_im = vy_max - vIdy * vdy;

    double8 re = (double8)(0);
    double8 im = (double8)(0);
    double8 re_2 = (double8)(0);
    double8 im_2 = (double8)(0);

    int8 counter = (int8)(0);
    int8 which = (int8)(-1);
    ushort i;
        
    for(i = 0; i < max_iter && any(which); i++) 
        {
        im = re*im;
        im = 2*im + c_im;
        re = re_2 - im_2 + c_re;
        re_2 = re*re;
        im_2 = im*im;
        
        which = convert_int8(re_2 + im_2 < fours);
        counter -= which;      
        }
    vstore8(convert_ushort8(counter % color), 0, output + Index);
}