#pragma once

// Header-only NumPy v1.0 (.npy) reader. Little-endian host only (x86).
//
// Self-test: a 1x3x375 int64 array saved with
//   np.save("t.npy", np.arange(1125, dtype=np.int64).reshape(1,3,375))
// loads as dtype 'q', shape {1,3,375}, npy_numel == 1125, i64[374] == 374.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct NpyArray {
    std::vector<int64_t> shape;
    std::vector<float>   f32;
    std::vector<int64_t> i64;
    std::vector<int32_t> i32;
    char dtype; // 'f'=f32, 'q'=i64, 'i'=i32
};

static size_t npy_numel(const NpyArray& a) {
    size_t n = a.shape.empty() ? 0 : 1;
    for (int64_t d : a.shape) n *= (size_t)d;
    return n;
}

[[noreturn]] static void npy_die(const char* msg, const char* path) {
    fprintf(stderr, "npy_load: %s (%s)\n", msg, path ? path : "?");
    exit(1);
}

static NpyArray npy_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) npy_die("cannot open file", path);

    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8) { fclose(f); npy_die("short read on magic", path); }
    if (memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); npy_die("bad magic", path); }
    // magic[6]=major, magic[7]=minor (v1.0 -> uint16 header len)

    unsigned char lenbytes[2];
    if (fread(lenbytes, 1, 2, f) != 2) { fclose(f); npy_die("short read on header len", path); }
    uint16_t hlen = (uint16_t)(lenbytes[0] | (lenbytes[1] << 8));

    std::string header(hlen, '\0');
    if (fread(&header[0], 1, hlen, f) != hlen) { fclose(f); npy_die("short read on header", path); }

    NpyArray arr;

    // descr
    {
        size_t p = header.find("'descr'");
        if (p == std::string::npos) { fclose(f); npy_die("missing descr", path); }
        size_t q = header.find('\'', header.find(':', p) + 1);
        if (q == std::string::npos) { fclose(f); npy_die("malformed descr", path); }
        size_t e = header.find('\'', q + 1);
        std::string descr = header.substr(q + 1, e - (q + 1));
        if (descr == "<f4")      arr.dtype = 'f';
        else if (descr == "<i8") arr.dtype = 'q';
        else if (descr == "<i4") arr.dtype = 'i';
        else { fclose(f); npy_die(("unsupported dtype: " + descr).c_str(), path); }
    }

    // fortran_order
    bool fortran = false;
    {
        size_t p = header.find("'fortran_order'");
        if (p == std::string::npos) { fclose(f); npy_die("missing fortran_order", path); }
        size_t t = header.find("True", p);
        size_t fa = header.find("False", p);
        size_t comma = header.find(',', p);
        if (t != std::string::npos && (fa == std::string::npos || t < fa) &&
            (comma == std::string::npos || t < comma)) {
            fortran = true;
        }
    }

    // shape
    {
        size_t p = header.find("'shape'");
        if (p == std::string::npos) { fclose(f); npy_die("missing shape", path); }
        size_t lp = header.find('(', p);
        size_t rp = header.find(')', lp);
        if (lp == std::string::npos || rp == std::string::npos) { fclose(f); npy_die("malformed shape", path); }
        std::string body = header.substr(lp + 1, rp - (lp + 1));
        size_t i = 0;
        while (i < body.size()) {
            while (i < body.size() && (body[i] == ' ' || body[i] == ',')) i++;
            if (i >= body.size()) break;
            size_t j = i;
            int64_t v = 0;
            while (j < body.size() && body[j] >= '0' && body[j] <= '9') {
                v = v * 10 + (body[j] - '0');
                j++;
            }
            if (j > i) arr.shape.push_back(v);
            i = (j > i) ? j : i + 1;
        }
    }

    size_t n = npy_numel(arr);
    bool ok = true;
    if (arr.dtype == 'f') {
        arr.f32.resize(n);
        ok = (n == 0) || (fread(arr.f32.data(), sizeof(float), n, f) == n);
    } else if (arr.dtype == 'q') {
        arr.i64.resize(n);
        ok = (n == 0) || (fread(arr.i64.data(), sizeof(int64_t), n, f) == n);
    } else { // 'i'
        arr.i32.resize(n);
        ok = (n == 0) || (fread(arr.i32.data(), sizeof(int32_t), n, f) == n);
    }
    fclose(f);
    if (!ok) npy_die("short read on data", path);

    // Fortran (column-major) -> C (row-major) reorder so downstream sees standard layout.
    if (fortran && arr.shape.size() > 1 && n > 1) {
        int    nd = (int) arr.shape.size();
        size_t elem = arr.dtype == 'q' ? sizeof(int64_t) : (arr.dtype == 'i' ? sizeof(int32_t) : sizeof(float));
        const unsigned char * src = arr.dtype == 'q' ? (const unsigned char *) arr.i64.data()
                                    : arr.dtype == 'i' ? (const unsigned char *) arr.i32.data()
                                                       : (const unsigned char *) arr.f32.data();
        std::vector<unsigned char> tmp((size_t) n * elem);
        std::vector<size_t> cstr(nd), fstr(nd);  // C and Fortran strides (in elements)
        size_t acc = 1;
        for (int d = nd - 1; d >= 0; d--) { cstr[d] = acc; acc *= (size_t) arr.shape[d]; }
        acc = 1;
        for (int d = 0; d < nd; d++) { fstr[d] = acc; acc *= (size_t) arr.shape[d]; }
        std::vector<size_t> idx(nd, 0);
        for (size_t lin = 0; lin < n; lin++) {
            size_t rem = lin;
            for (int d = 0; d < nd; d++) { idx[d] = rem / cstr[d]; rem %= cstr[d]; }
            size_t fpos = 0;
            for (int d = 0; d < nd; d++) fpos += idx[d] * fstr[d];
            memcpy(&tmp[lin * elem], &src[fpos * elem], elem);
        }
        if (arr.dtype == 'q')      memcpy(arr.i64.data(), tmp.data(), tmp.size());
        else if (arr.dtype == 'i') memcpy(arr.i32.data(), tmp.data(), tmp.size());
        else                       memcpy(arr.f32.data(), tmp.data(), tmp.size());
    }
    return arr;
}

static bool npy_save_i64(const char * path, const int64_t * data, const std::vector<int64_t> & shape) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        sh += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size()) sh += ",";
        if (i + 1 < shape.size()) sh += " ";
    }
    sh += ")";
    std::string hdr = "{'descr': '<i8', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t total = 10 + hdr.size() + 1;  // magic+ver+len(2) + header + newline
    size_t pad   = (64 - (total % 64)) % 64;
    hdr.append(pad, ' ');
    hdr += '\n';
    uint16_t hlen = (uint16_t) hdr.size();
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    unsigned char lb[2] = { (unsigned char) (hlen & 0xff), (unsigned char) (hlen >> 8) };
    fwrite(lb, 1, 2, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    size_t n = shape.empty() ? 0 : 1;
    for (int64_t d : shape) n *= (size_t) d;
    bool ok = (n == 0) || (fwrite(data, sizeof(int64_t), n, f) == n);
    fclose(f);
    return ok;
}
