using Polynomials

function popov_hopping(dR)
#function hoppingInter(dR)
    ang_per_bohr = 0.529177 # [Anstroms/Bohr radius]
    eV_per_hart = 27.2114 # [eV/Hartree]
    dR = dR / ang_per_bohr
    dRn = dR / norm(dR)

    l = dRn[1]
    m = dRn[2]
    n = dRn[3]
    r = norm(dR)
    if r > 10 || r < 1
        return 0
    end


    aa = 1.0 #[Bohr radii]
    b = 10.0 #[Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    Css_sigma = [-0.5286482, 0.4368816, -0.2390807, 0.0701587,
                 0.0106355, -0.0258943, 0.0169584, -0.0070929,
                 0.0019797, -0.000304]
    Csp_sigma = [0.3865122, -0.2909735, 0.1005869, 0.0340820,
                 -0.0705311, 0.0528565, -0.0270332, 0.0103844,
                 -0.0028724, 0.0004584]
    Cpp_sigma = [0.1727212, -0.0937225, -0.0445544, 0.1114266,
                 -0.0978079, 0.0577363, -0.0262833, 0.0094388,
                 -0.0024695, 0.0003863]
    Cpp_pi = [-0.3969243, 0.3477657, -0.2357499, 0.1257478,
              -0.0535682, 0.0181983, -0.0046855, 0.0007303,
              0.0000225, -0.0000393]
    Vss_sigma = ChebyshevT(Css_sigma)(y)
    Vsp_sigma = ChebyshevT(Csp_sigma)(y)
    Vpp_sigma = ChebyshevT(Cpp_sigma)(y)
    Vpp_pi = ChebyshevT(Cpp_pi)(y)

    Vss_sigma -= Css_sigma[1] / 2
    Vsp_sigma -= Csp_sigma[1] / 2
    Vpp_sigma -= Cpp_sigma[1] / 2
    Vpp_pi -= Cpp_pi[1] / 2
    Ezz = n^2 * Vpp_sigma + (1 - n^2) * Vpp_pi
    valmat = Ezz
    return valmat * eV_per_hart
end

function overlapInter(dR)
    ang_per_bohr = 0.529177 # [Anstroms/Bohr radius]
    eV_per_hart = 27.2114 # [eV/Hartree]
    dR = dR / ang_per_bohr
    dRn = dR / norm(dR)

    l = dRn[1]
    m = dRn[2]
    n = dRn[3]
    r = norm(dR)
    if r > 10 || r < 1
        return 0
    end

    aa = 1.0 #[Bohr radii]
    b = 10.0 #[Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    Css_sigma = [0.4524096, -0.3678693, 0.1903822, -0.0484968,
                 -0.0099673, 0.0153765, -0.0071442, 0.0017435,
                 -0.0001224, -0.0000443]
    Csp_sigma = [-0.3509680, 0.2526017, -0.0661301, -0.0465212,
                 0.0572892, -0.0289944, 0.0078424, -0.0004892,
                 -0.0004677, 0.0001590]
    Cpp_sigma = [-0.0571487, -0.0291832, 0.1558650, -0.1665997,
                 0.0921727, -0.0268106, 0.0002240, 0.0040319,
                 -0.0022450, 0.0005596]
    Cpp_pi = [0.3797305, -0.3199876, 0.1897988, -0.0754124,
              0.0156376, 0.0025976, -0.0039498, 0.0020581,
              -0.0007114, 0.0001427]
    Vss_sigma = ChebyshevT(Css_sigma)(y)
    Vsp_sigma = ChebyshevT(Csp_sigma)(y)
    Vpp_sigma = ChebyshevT(Cpp_sigma)(y)
    Vpp_pi = ChebyshevT(Cpp_pi)(y)

    Vss_sigma -= Css_sigma[1] / 2
    Vsp_sigma -= Csp_sigma[1] / 2
    Vpp_sigma -= Cpp_sigma[1] / 2
    Vpp_pi -= Cpp_pi[1] / 2
    Ezz = n^2 * Vpp_sigma + (1 - n^2) * Vpp_pi
    valmat = Ezz
    return valmat * eV_per_hart
end

function porezag_hopping(dR)
#function hoppingIntra(dR)
    ang_per_bohr = 0.529177 # [Anstroms/Bohr radius]
    eV_per_hart = 27.2114 # [eV/Hartree]
    dR = dR / ang_per_bohr
    dRn = dR / norm(dR)

    l = dRn[1]
    m = dRn[2]
    n = dRn[3]
    r = norm(dR)
    if r > 7 || r < 1
        return 0
    end

    aa = 1.0 #[Bohr radii]
    b = 7.0 #[Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    Css_sigma = [-0.4663805, 0.3528951, -0.1402985, 0.0050519,
                 0.0269723, -0.0158810, 0.0036716, 0.0010301,
                 -0.0015546, 0.0008601]
    Csp_sigma = [0.3395418, -0.2250358, 0.0298224, 0.0653476,
                 -0.0605786, 0.0298962, -0.0099609, 0.0020609,
                 0.0001264, -0.0003381]
    Cpp_sigma = [0.2422701, -0.1315258, -0.0372696, 0.0942352,
                 -0.0673216, 0.0316900, -0.0117293, 0.0033519,
                 -0.0004838, -0.0000906]
    Cpp_pi = [-0.3793837, 0.3204470, -0.1956799, 0.0883986,
              -0.0300733, 0.0074465, -0.0008563, -0.0004453,
              0.0003842, -0.0001855]
    Vss_sigma = ChebyshevT(Css_sigma)(y)
    Vsp_sigma = ChebyshevT(Csp_sigma)(y)
    Vpp_sigma = ChebyshevT(Cpp_sigma)(y)
    Vpp_pi = ChebyshevT(Cpp_pi)(y)

    Vss_sigma -= Css_sigma[1] / 2
    Vsp_sigma -= Csp_sigma[1] / 2
    Vpp_sigma -= Cpp_sigma[1] / 2
    Vpp_pi -= Cpp_pi[1] / 2
    Ezz = n^2 * Vpp_sigma + (1 - n^2) * Vpp_pi
    valmat = Ezz
    return valmat * eV_per_hart
end

function overlapIntra(dR)
    ang_per_bohr = 0.529177 # [Anstroms/Bohr radius]
    eV_per_hart = 27.2114 # [eV/Hartree]
    dR = dR / ang_per_bohr
    dRn = dR / norm(dR)

    l = dRn[1]
    m = dRn[2]
    n = dRn[3]
    r = norm(dR)
    if r > 7 || r < 1
        return 0
    end

    aa = 1.0 #[Bohr radii]
    b = 7.0 #[Bohr radii]
    y = (2.0 * r - (b + aa)) / (b - aa)

    Css_sigma = [0.4728644, -0.3661623, 0.1594782, -0.0204934,
                 -0.0170732, 0.0096695, -0.0007135, -0.0013826,
                 0.0007849, -0.0002005]
    Csp_sigma = [-0.3662838, 0.2490285, -0.0431248, -0.0584391,
                 0.0492775, -0.0150447, -0.0010758, 0.0027734,
                 -0.0011214, 0.0002303]
    Cpp_sigma = [-0.1359608, 0.0226235, 0.1406440, -0.1573794,
                 0.0753818, -0.0108677, -0.0075444, 0.0051533,
                 -0.0013747, 0.0000751]
    Cpp_pi = [0.3715732, -0.3070867, 0.1707304, -0.0581555,
              0.0061645, 0.0051460, -0.0032776, 0.0009119,
              -0.0001265, -0.000227]
    Vss_sigma = ChebyshevT(Css_sigma)(y)
    Vsp_sigma = ChebyshevT(Csp_sigma)(y)
    Vpp_sigma = ChebyshevT(Cpp_sigma)(y)
    Vpp_pi = ChebyshevT(Cpp_pi)(y)

    Vss_sigma -= Css_sigma[1] / 2
    Vsp_sigma -= Csp_sigma[1] / 2
    Vpp_sigma -= Cpp_sigma[1] / 2
    Vpp_pi -= Cpp_pi[1] / 2
    Ezz = n^2 * Vpp_sigma + (1 - n^2) * Vpp_pi
    valmat = Ezz
    return valmat * eV_per_hart
end

function nnhop(dR)
    dist = norm(dR)-2
    return 1 - dist
end

function nnhop_intra(dR)
    return 2.7
end

function nnhop_inter(dR)
    return .3
end
