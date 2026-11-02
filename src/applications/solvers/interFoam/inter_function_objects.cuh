#pragma once
// inter_function_objects.cuh -- the runtime function objects of interFoam's controlDict `functions { }`.
//
// brae ran none of them: the writer said so in one notice and the case's postProcessing/ never appeared.
// This file runs the subset a refill/slosh study reads its answers from, with OpenFOAM's own semantics and
// OpenFOAM's own file layout, so the usual post-processing scripts read brae's output unchanged:
//
//   volFieldValue      regionType all | cellZone,  operation volIntegrate | volAverage | sum | min | max
//   surfaceFieldValue  regionType patch,           operation areaAverage | areaIntegrate | sum | min | max
//                      (the phase fraction's patch values; no other field has patch values on the host)
//   probes             fixedLocations, cell values (interpolationScheme cell) of the phase fraction, U, p_rgh
//   fieldMinMax        mode magnitude | component, of U and of any scalar field
//   exprField          action new, a scalar expression of the phase fraction, p_rgh, U (.x() .y() .z(),
//                      mag, magSqr), pos().x()/y()/z(), other expression fields and numbers
//   runTimeControl     conditions of type minMax on another object's result, satisfiedAction end
//
// WHEN THEY RUN. At the end of each time step, in the controlDict's order -- OpenFOAM runs them at the top
// of the next Time::run(), on the same state. An object's `writeControl` (timeStep | runTime |
// adjustableRunTime | writeTime | none) and `writeInterval` say when it computes and writes; there is no
// output at the start time, as in OpenFOAM (the execution index starts at 0). runTimeControl evaluates every
// step. `enabled false` removes an object, as in OpenFOAM, and `foamDictionary -entry
// functions.<name>.enabled -set false` between two runs therefore works as it does there.
//
// WHERE THE STATE COMES FROM. The driver hands over lazy host views (Source): a loop that keeps its fields on
// the device downloads a field the first time an object asks for it in a step, and not at all in a step in
// which nothing is due. Expression fields are evaluated on demand, from those views, once per step.
//
// WHAT IS NOT HERE, said rather than guessed: an entry of a ported type that asks for something unported
// (writeFields, weightField, a vector field in volFieldValue, interpolationScheme cellPoint, exprField
// autowrite, a minMax mode other than maximum/minimum ...) stops the run at start-up and names itself. An
// object of a type that is not ported at all (surfaces, forces, fieldAverage ...) is noticed and left alone,
// as before.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "brae_notice.cuh"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {
namespace fo {

// ---------------------------------------------------------------------------------------------------------------
// what an object may read, per step
// ---------------------------------------------------------------------------------------------------------------

struct Step
{
    label  timeIndex = 0;        // the absolute index, as Time::timeIndex() (a restart continues it)
    scalar time = 0;
    scalar deltaT = 0;
    scalar sinceStart = 0;       // time - startTime
    bool   runTimeWrite = false; // the run writes its fields at this step (writeControl writeTime)
};

// Host views of the solution at the step just taken. Every getter may download on first use; `newStep` is
// called once at the start of each execute() so the driver can drop what it cached.
struct Source
{
    std::function<void()>                                       newStep;
    std::function<const std::vector<scalar>&()>                 alpha;
    std::function<const std::vector<scalar>&()>                 p_rgh;
    std::function<const std::vector<scalar>&()>                 Ux;
    std::function<const std::vector<scalar>&()>                 Uy;
    std::function<const std::vector<scalar>&()>                 Uz;
    std::function<const std::vector<scalar>&(std::size_t)>      alphaPatch;   // the phase fraction's patch values
};

struct Services
{
    std::string                                        caseDir;
    std::string                                        startName;     // postProcessing/<object>/<startName>/
    int                                                precision = 6; // controlDict writePrecision
    std::string                                        phaseField = "alpha.water";
    const PrimitiveMesh*                               mesh = nullptr;
    const FvGeometry*                                  geom = nullptr;
    const std::vector<FvPatch>*                        patches = nullptr;
    const std::map<std::string, std::vector<label>>*   cellZones = nullptr;
    Source                                             src;
};

// ---------------------------------------------------------------------------------------------------------------
// scalar / vector field values and the expression evaluator
// ---------------------------------------------------------------------------------------------------------------

// A scalar field that is a constant, a borrowed host vector or an owned one.
struct SF
{
    scalar                               c = 0;
    const std::vector<scalar>*           ref = nullptr;
    std::shared_ptr<std::vector<scalar>> own;
    bool isConst() const { return !ref && !own; }
    const std::vector<scalar>& v() const { return own ? *own : *ref; }
};

struct VF { SF x, y, z; };

struct Val
{
    bool vec = false;
    SF   s;
    VF   v;
};

inline SF sfConst(scalar c) { SF r; r.c = c; return r; }
inline SF sfBorrow(const std::vector<scalar>& v) { SF r; r.ref = &v; return r; }

inline SF sfBinary(const SF& a, const SF& b, char op, std::size_t n)
{
    auto apply = [op](scalar x, scalar y) -> scalar
    {
        switch (op)
        {
            case '+': return x + y;
            case '-': return x - y;
            case '*': return x*y;
            default:  return x/y;
        }
    };
    if (a.isConst() && b.isConst())
    {
        return sfConst(apply(a.c, b.c));
    }
    auto out = std::make_shared<std::vector<scalar>>(n);
    const scalar* pa = a.isConst() ? &a.c : a.v().data();
    const scalar* pb = b.isConst() ? &b.c : b.v().data();
    const std::size_t sa = a.isConst() ? 0 : 1;
    const std::size_t sb = b.isConst() ? 0 : 1;
    scalar* po = out->data();
    switch (op)
    {
        case '+': for (std::size_t i = 0; i < n; ++i) po[i] = pa[i*sa] + pb[i*sb]; break;
        case '-': for (std::size_t i = 0; i < n; ++i) po[i] = pa[i*sa] - pb[i*sb]; break;
        case '*': for (std::size_t i = 0; i < n; ++i) po[i] = pa[i*sa]*pb[i*sb];   break;
        default:  for (std::size_t i = 0; i < n; ++i) po[i] = pa[i*sa]/pb[i*sb];   break;
    }
    SF r;
    r.own = out;
    return r;
}

template <class F>
inline SF sfMap(const SF& a, F f, std::size_t n)
{
    if (a.isConst())
    {
        return sfConst(f(a.c));
    }
    auto out = std::make_shared<std::vector<scalar>>(n);
    const scalar* pa = a.v().data();
    for (std::size_t i = 0; i < n; ++i)
    {
        (*out)[i] = f(pa[i]);
    }
    SF r;
    r.own = out;
    return r;
}

// What the expression evaluator may name.
struct Scope
{
    std::size_t                                          n = 0;
    std::function<bool(const std::string&, SF&)>         scalarField;
    std::function<bool(const std::string&, VF&)>         vectorField;
    std::function<VF()>                                  pos;
};

struct Tok
{
    enum Kind { Num, Id, Op, End } k = End;
    std::string t;
    scalar      num = 0;
};

inline std::vector<Tok> tokenizeExpression(const std::string& s)
{
    std::vector<Tok> out;
    std::size_t i = 0;
    while (i < s.size())
    {
        const char ch = s[i];
        if (std::isspace(static_cast<unsigned char>(ch)))
        {
            ++i;
            continue;
        }
        const bool afterClose = !out.empty() && out.back().k == Tok::Op && out.back().t == ")";
        if (std::isdigit(static_cast<unsigned char>(ch))
         || (ch == '.' && i + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[i + 1])) && !afterClose))
        {
            char* end = nullptr;
            const scalar v = std::strtod(s.c_str() + i, &end);
            Tok t;
            t.k = Tok::Num;
            t.num = v;
            t.t = s.substr(i, static_cast<std::size_t>(end - (s.c_str() + i)));
            i = static_cast<std::size_t>(end - s.c_str());
            out.push_back(t);
        }
        else if (std::isalpha(static_cast<unsigned char>(ch)) || ch == '_')
        {
            std::size_t j = i;
            while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_'))
            {
                ++j;
            }
            Tok t;
            t.k = Tok::Id;
            t.t = s.substr(i, j - i);
            out.push_back(t);
            i = j;
        }
        else if (std::strchr("+-*/(),.", ch))
        {
            Tok t;
            t.k = Tok::Op;
            t.t = std::string(1, ch);
            out.push_back(t);
            ++i;
        }
        else
        {
            throw std::runtime_error(std::string("brae function objects: the expression `") + s
                                     + "` contains `" + ch + "`, which the expression evaluator does not read");
        }
    }
    out.push_back(Tok());   // End
    return out;
}

class ExprParser
{
public:
    ExprParser(const std::string& text, Scope& sc)
      : text_(text), t_(tokenizeExpression(text)), sc_(sc)
    {}

    SF parseScalar()
    {
        Val v = expr();
        if (peek().k != Tok::End)
        {
            fail("unexpected `" + peek().t + "`");
        }
        if (v.vec)
        {
            fail("it is a vector; an exprField here is a scalar field");
        }
        return v.s;
    }

private:
    std::string       text_;
    std::vector<Tok>  t_;
    std::size_t       p_ = 0;
    Scope&            sc_;

    [[noreturn]] void fail(const std::string& why) const
    {
        throw std::runtime_error("brae function objects: expression `" + text_ + "`: " + why);
    }
    const Tok& peek(std::size_t o = 0) const { return t_[std::min(p_ + o, t_.size() - 1)]; }
    bool isOp(const char* s, std::size_t o = 0) const { return peek(o).k == Tok::Op && peek(o).t == s; }
    void expectOp(const char* s)
    {
        if (!isOp(s))
        {
            fail(std::string("expected `") + s + "`");
        }
        ++p_;
    }

    Val scalarVal(const SF& s) const { Val v; v.s = s; return v; }
    const SF& needScalar(const Val& v) const
    {
        if (v.vec)
        {
            fail("vector arithmetic is not supported; use .x() .y() .z(), mag() or magSqr()");
        }
        return v.s;
    }

    Val expr()
    {
        Val l = term();
        while (isOp("+") || isOp("-"))
        {
            const char op = t_[p_++].t[0];
            Val r = term();
            l = scalarVal(sfBinary(needScalar(l), needScalar(r), op, sc_.n));
        }
        return l;
    }
    Val term()
    {
        Val l = unary();
        while (isOp("*") || isOp("/"))
        {
            const char op = t_[p_++].t[0];
            Val r = unary();
            l = scalarVal(sfBinary(needScalar(l), needScalar(r), op, sc_.n));
        }
        return l;
    }
    Val unary()
    {
        if (isOp("-"))
        {
            ++p_;
            Val v = unary();
            return scalarVal(sfMap(needScalar(v), [](scalar x) { return -x; }, sc_.n));
        }
        if (isOp("+"))
        {
            ++p_;
            return unary();
        }
        return primary();
    }

    static int componentIndex(const std::string& s)
    {
        return s == "x" ? 0 : s == "y" ? 1 : s == "z" ? 2 : -1;
    }
    Val component(const Val& v, int c) const
    {
        if (!v.vec)
        {
            fail("a component of a scalar");
        }
        return scalarVal(c == 0 ? v.v.x : c == 1 ? v.v.y : v.v.z);
    }
    Val postfix(Val v)
    {
        while (isOp(".") && peek(1).k == Tok::Id && componentIndex(peek(1).t) >= 0 && isOp("(", 2) && isOp(")", 3))
        {
            const int c = componentIndex(peek(1).t);
            p_ += 4;
            v = component(v, c);
        }
        return v;
    }

    Val callFunction(const std::string& name, const std::vector<Val>& args)
    {
        const std::size_t n = sc_.n;
        auto sq = [](scalar x) { return x*x; };
        if (name == "pos")
        {
            if (!args.empty())
            {
                fail("pos() takes no argument");
            }
            Val v;
            v.vec = true;
            v.v = sc_.pos();
            return v;
        }
        if (args.size() != 1)
        {
            fail("function `" + name + "` is not supported here (mag, magSqr, sqr, sqrt and pos() are)");
        }
        const Val& a = args[0];
        if (name == "magSqr" || name == "mag")
        {
            SF s;
            if (a.vec)
            {
                s = sfBinary(sfBinary(sfMap(a.v.x, sq, n), sfMap(a.v.y, sq, n), '+', n), sfMap(a.v.z, sq, n), '+', n);
            }
            else
            {
                s = sfMap(a.s, sq, n);
            }
            return scalarVal(name == "mag" ? sfMap(s, [](scalar x) { return std::sqrt(x); }, n) : s);
        }
        if (name == "sqr")
        {
            return scalarVal(sfMap(needScalar(a), sq, n));
        }
        if (name == "sqrt")
        {
            return scalarVal(sfMap(needScalar(a), [](scalar x) { return std::sqrt(x); }, n));
        }
        fail("function `" + name + "` is not supported here (mag, magSqr, sqr, sqrt and pos() are)");
    }

    Val primary()
    {
        const Tok tk = peek();
        if (tk.k == Tok::Num)
        {
            ++p_;
            return scalarVal(sfConst(tk.num));
        }
        if (isOp("("))
        {
            ++p_;
            Val v = expr();
            expectOp(")");
            return postfix(v);
        }
        if (tk.k != Tok::Id)
        {
            fail(tk.k == Tok::End ? "it ends early" : "unexpected `" + tk.t + "`");
        }
        std::vector<std::string> parts;
        parts.push_back(tk.t);
        ++p_;
        while (isOp(".") && peek(1).k == Tok::Id && !(componentIndex(peek(1).t) >= 0 && isOp("(", 2)))
        {
            parts.push_back(peek(1).t);
            p_ += 2;
        }
        if (parts.size() == 1 && isOp("("))
        {
            ++p_;
            std::vector<Val> args;
            if (!isOp(")"))
            {
                args.push_back(expr());
                while (isOp(","))
                {
                    ++p_;
                    args.push_back(expr());
                }
            }
            expectOp(")");
            return postfix(callFunction(parts[0], args));
        }
        std::string name = parts[0];
        for (std::size_t i = 1; i < parts.size(); ++i)
        {
            name += "." + parts[i];
        }
        SF s;
        if (sc_.scalarField(name, s))
        {
            return scalarVal(s);
        }
        VF vf;
        if (sc_.vectorField(name, vf))
        {
            Val v;
            v.vec = true;
            v.v = vf;
            return postfix(v);
        }
        fail("`" + name + "` is neither a field nor a function here");
    }
};

// ---------------------------------------------------------------------------------------------------------------
// controls and files
// ---------------------------------------------------------------------------------------------------------------

// Foam::timeControl::execute() for the write cadences brae supports.
struct Control
{
    enum Kind { TimeStep, RunTime, Adjustable, WriteTime, None } kind = TimeStep;
    scalar interval = 1;
    label  executionIndex = 0;

    static Control read(const FoamDict& d, const std::string& object, const std::string& prefix)
    {
        Control c;
        const std::string w = d.wordOr(prefix + "Control", "timeStep");
        if (w == "timeStep")             c.kind = TimeStep;
        else if (w == "runTime")         c.kind = RunTime;
        else if (w == "adjustableRunTime") c.kind = Adjustable;
        else if (w == "writeTime" || w == "outputTime") c.kind = WriteTime;
        else if (w == "none")            c.kind = None;
        else
        {
            throw std::runtime_error("brae function objects: `" + object + "` has " + prefix + "Control " + w
                                     + "; brae has timeStep, runTime, adjustableRunTime, writeTime and none");
        }
        c.interval = d.scalarOr(prefix + "Interval", 1);
        if ((c.kind == TimeStep || c.kind == RunTime || c.kind == Adjustable) && !(c.interval > 0))
        {
            throw std::runtime_error("brae function objects: `" + object + "` has " + prefix + "Interval "
                                     + std::to_string(c.interval) + ", which is not positive");
        }
        return c;
    }

    bool due(const Step& s)
    {
        switch (kind)
        {
            case TimeStep:
            {
                const label n = std::max<label>(1, static_cast<label>(interval));
                return (s.timeIndex % n) == 0;
            }
            case RunTime:
            case Adjustable:
            {
                const label idx = static_cast<label>((s.sinceStart + scalar(0.5)*s.deltaT)/interval);
                if (idx > executionIndex)
                {
                    executionIndex = idx;
                    return true;
                }
                return false;
            }
            case WriteTime:
                return s.runTimeWrite;
            default:
                return false;
        }
    }
};

inline std::string fmtNum(scalar v, int precision)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*g", precision, v);
    return buf;
}

// postProcessing/<object>/<startTime>/<file>, created with its first row, flushed after every row so a
// running case can be read by a script while it runs.
class DatFile
{
public:
    DatFile() = default;
    DatFile(std::string dir, std::string file) : dir_(std::move(dir)), file_(std::move(file)) {}

    void header(const std::string& line) { head_.push_back(line); }
    void row(const std::string& line)
    {
        if (!os_.is_open())
        {
            std::filesystem::create_directories(dir_);
            os_.open(dir_ + "/" + file_, std::ios::out | std::ios::trunc);
            if (!os_)
            {
                throw std::runtime_error("brae function objects: cannot write " + dir_ + "/" + file_);
            }
            for (const std::string& h : head_)
            {
                os_ << h << '\n';
            }
        }
        os_ << line << '\n';
        os_.flush();
    }

private:
    std::string               dir_, file_;
    std::vector<std::string>  head_;
    std::ofstream             os_;
};

// ---------------------------------------------------------------------------------------------------------------
// the run-time state every object shares
// ---------------------------------------------------------------------------------------------------------------

class Runtime
{
public:
    explicit Runtime(Services s) : svc(std::move(s)) {}

    Services svc;
    label    stamp = 0;                                                  // one per execute()
    std::map<std::string, std::string>                    exprs;         // exprField name -> expression
    std::map<std::string, std::map<std::string, scalar>> results;        // object -> "operation(field)" -> value

    std::size_t nCells() const { return svc.geom->V().size(); }

    std::string dir(const std::string& object) const
    {
        return svc.caseDir + "/postProcessing/" + object + "/" + svc.startName;
    }

    bool isScalarName(const std::string& name) const
    {
        return name == svc.phaseField || name == "p_rgh" || exprs.count(name) > 0;
    }

    SF scalarField(const std::string& name)
    {
        SF s;
        if (!lookupScalar(name, s))
        {
            throw std::runtime_error("brae function objects: the field `" + name + "` is not one brae can give a "
                                     "function object (" + svc.phaseField + ", p_rgh, U and exprField results are)");
        }
        return s;
    }

    VF velocity()
    {
        VF v;
        v.x = sfBorrow(svc.src.Ux());
        v.y = sfBorrow(svc.src.Uy());
        v.z = sfBorrow(svc.src.Uz());
        return v;
    }

    VF cellCentres()
    {
        if (!posX_)
        {
            const std::vector<vector>& C = svc.geom->C();
            posX_ = std::make_shared<std::vector<scalar>>(C.size());
            posY_ = std::make_shared<std::vector<scalar>>(C.size());
            posZ_ = std::make_shared<std::vector<scalar>>(C.size());
            for (std::size_t i = 0; i < C.size(); ++i)
            {
                (*posX_)[i] = C[i].x;
                (*posY_)[i] = C[i].y;
                (*posZ_)[i] = C[i].z;
            }
        }
        VF v;
        v.x.ref = posX_.get();
        v.y.ref = posY_.get();
        v.z.ref = posZ_.get();
        return v;
    }

    scalar result(const std::string& object, const std::string& key) const
    {
        const auto o = results.find(object);
        if (o == results.end() || !o->second.count(key))
        {
            throw std::runtime_error("brae function objects: `" + key + "` of object `" + object + "` is not "
                                     "available at this step (the object is absent, disabled or not due yet)");
        }
        return o->second.at(key);
    }

private:
    std::map<std::string, std::pair<label, SF>> cache_;
    std::vector<std::string>                    active_;
    std::shared_ptr<std::vector<scalar>>        posX_, posY_, posZ_;

    bool lookupScalar(const std::string& name, SF& out)
    {
        if (name == svc.phaseField)
        {
            out = sfBorrow(svc.src.alpha());
            return true;
        }
        if (name == "p_rgh")
        {
            out = sfBorrow(svc.src.p_rgh());
            return true;
        }
        const auto e = exprs.find(name);
        if (e == exprs.end())
        {
            return false;
        }
        const auto c = cache_.find(name);
        if (c != cache_.end() && c->second.first == stamp)
        {
            out = c->second.second;
            return true;
        }
        if (std::find(active_.begin(), active_.end(), name) != active_.end())
        {
            throw std::runtime_error("brae function objects: the expression fields refer to each other in a "
                                     "loop at `" + name + "`");
        }
        active_.push_back(name);
        Scope sc;
        sc.n = nCells();
        sc.scalarField = [this](const std::string& nm, SF& o) { return lookupScalar(nm, o); };
        sc.vectorField = [this](const std::string& nm, VF& o)
        {
            if (nm != "U")
            {
                return false;
            }
            o = velocity();
            return true;
        };
        sc.pos = [this]() { return cellCentres(); };
        SF r;
        try
        {
            r = ExprParser(e->second, sc).parseScalar();
        }
        catch (...)
        {
            active_.pop_back();
            throw;
        }
        active_.pop_back();
        cache_[name] = std::make_pair(stamp, r);
        out = r;
        return true;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// the objects
// ---------------------------------------------------------------------------------------------------------------

class Object
{
public:
    virtual ~Object() = default;
    std::string name;
    bool        enabled = true;
    Control     ctl;
    bool        log = true;

    // compute and write, when ctl says it is due
    virtual void write(const Step&, Runtime&) = 0;
    // every step, after the writes; true asks the run to end (and write) at this step
    virtual bool evaluatesEveryStep() const { return false; }
    virtual bool evaluate(const Step&, Runtime&) { return false; }
};

inline void refuseKey(const FoamDict& d, const std::string& object, const char* key, const char* why)
{
    if (d.found(key))
    {
        throw std::runtime_error("brae function objects: `" + object + "` sets `" + key + "` -- " + why);
    }
}

inline std::vector<std::string> readFieldNames(const FoamDict& d, const std::string& key)
{
    // `fields (alpha.water alphaUz)` and `fields (volIntegrate(alphaUz))`: split at the top level only
    const auto* t = d.find(key);
    std::vector<std::string> out;
    if (!t)
    {
        return out;
    }
    // The reader hands parentheses over as tokens of their own, so `volIntegrate(alphaUz)` is three tokens.
    // A word followed by `(` is a call and stays one word; words are otherwise separated by a space.
    std::string joined;
    for (const std::string& s : *t)
    {
        const bool paren = (s == "(" || s == ")");
        if (paren)
        {
            joined += s;
        }
        else
        {
            joined += ((joined.empty() || joined.back() == '(') ? "" : " ") + s;
        }
    }
    const std::size_t a = joined.find('(');
    const std::size_t b = joined.rfind(')');
    if (a == std::string::npos || b == std::string::npos || b < a)
    {
        throw std::runtime_error("brae function objects: `" + key + "` is not a list");
    }
    std::string cur;
    int depth = 0;
    for (std::size_t i = a + 1; i < b; ++i)
    {
        const char ch = joined[i];
        if (ch == '(')
        {
            ++depth;
        }
        else if (ch == ')')
        {
            --depth;
        }
        if (depth == 0 && std::isspace(static_cast<unsigned char>(ch)))
        {
            if (!cur.empty())
            {
                out.push_back(cur);
                cur.clear();
            }
        }
        else if (!std::isspace(static_cast<unsigned char>(ch)))
        {
            cur += ch;
        }
    }
    if (!cur.empty())
    {
        out.push_back(cur);
    }
    return out;
}

// the region's cells (empty `all`) and the sums every volume operation needs
struct CellRegion
{
    bool                all = true;
    std::vector<label>  cells;
    std::string         description;
    scalar              volume = 0;
    label               n = 0;
};

inline CellRegion readCellRegion(const FoamDict& d, const std::string& object, Runtime& rt)
{
    CellRegion r;
    const std::string type = d.wordOr("regionType", "");
    const std::vector<scalar>& V = rt.svc.geom->V();
    if (type == "all")
    {
        r.all = true;
        r.description = "all";
        r.n = static_cast<label>(V.size());
        for (scalar v : V)
        {
            r.volume += v;
        }
    }
    else if (type == "cellZone")
    {
        const std::string zone = d.wordOr("name", "");
        const auto it = rt.svc.cellZones ? rt.svc.cellZones->find(zone) : std::map<std::string, std::vector<label>>::const_iterator();
        if (!rt.svc.cellZones || it == rt.svc.cellZones->end())
        {
            throw std::runtime_error("brae function objects: `" + object + "` names the cellZone `" + zone
                                     + "`, which is not in constant/polyMesh/cellZones");
        }
        r.all = false;
        r.cells = it->second;
        r.description = "cellZone " + zone;
        r.n = static_cast<label>(r.cells.size());
        for (label c : r.cells)
        {
            r.volume += V[static_cast<std::size_t>(c)];
        }
    }
    else
    {
        throw std::runtime_error("brae function objects: `" + object + "` has regionType `" + type
                                 + "`; brae has all and cellZone for volFieldValue");
    }
    return r;
}

class VolFieldValue : public Object
{
public:
    VolFieldValue(const std::string& nm, const FoamDict& d, Runtime& rt)
    {
        name = nm;
        regionType_ = d.wordOr("regionType", "");
        region_ = readCellRegion(d, name, rt);
        op_ = d.wordOr("operation", "");
        if (op_ != "volIntegrate" && op_ != "volAverage" && op_ != "sum" && op_ != "min" && op_ != "max")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has operation `" + op_
                                     + "`; brae has volIntegrate, volAverage, sum, min and max");
        }
        fields_ = readFieldNames(d, "fields");
        if (fields_.empty())
        {
            throw std::runtime_error("brae function objects: `" + name + "` lists no fields");
        }
        for (const std::string& f : fields_)
        {
            if (f == "U" || !rt.isScalarName(f))
            {
                throw std::runtime_error("brae function objects: `" + name + "` takes the field `" + f
                                         + "`, which is not a scalar field brae knows (" + rt.svc.phaseField
                                         + ", p_rgh and exprField results are)");
            }
        }
        writeToFile_ = d.switchOr("writeToFile", true);
        if (d.switchOr("writeFields", false))
        {
            throw std::runtime_error("brae function objects: `" + name + "` sets writeFields; brae does not write "
                                     "the region's fields");
        }
        refuseKey(d, name, "weightField", "weighted operations are not ported");
        refuseKey(d, name, "postOperation", "post operations are not ported");
    }

    void open(Runtime& rt)
    {
        if (!writeToFile_)
        {
            return;
        }
        dat_ = DatFile(rt.dir(name), "volFieldValue.dat");
        dat_.header("# Region type : " + region_.description);
        dat_.header("# Cells       : " + std::to_string(region_.n));
        dat_.header("# Volume      : " + fmtNum(region_.volume, rt.svc.precision));
        std::string h = "# Time       ";
        for (const std::string& f : fields_)
        {
            h += " " + op_ + "(" + f + ")";
        }
        dat_.header(h);
    }

    void write(const Step& s, Runtime& rt) override
    {
        const std::vector<scalar>& V = rt.svc.geom->V();
        std::string line = fmtNum(s.time, rt.svc.precision);
        for (const std::string& f : fields_)
        {
            const SF sf = rt.scalarField(f);
            const scalar value = reduce(sf, V);
            rt.results[name][op_ + "(" + f + ")"] = value;
            line += "\t" + fmtNum(value, rt.svc.precision);
        }
        if (writeToFile_)
        {
            dat_.row(line);
        }
    }

private:
    std::string               regionType_, op_;
    std::vector<std::string>  fields_;
    CellRegion                region_;
    bool                      writeToFile_ = true;
    DatFile                   dat_;

    scalar reduce(const SF& f, const std::vector<scalar>& V) const
    {
        auto at = [&](std::size_t c) { return f.isConst() ? f.c : f.v()[c]; };
        scalar acc = 0;
        scalar vol = 0;
        scalar mn = std::numeric_limits<scalar>::max();
        scalar mx = -std::numeric_limits<scalar>::max();
        auto visit = [&](std::size_t c)
        {
            const scalar x = at(c);
            acc += (op_ == "volIntegrate" || op_ == "volAverage") ? V[c]*x : x;
            vol += V[c];
            mn = std::min(mn, x);
            mx = std::max(mx, x);
        };
        if (region_.all)
        {
            for (std::size_t c = 0; c < V.size(); ++c) visit(c);
        }
        else
        {
            for (label c : region_.cells) visit(static_cast<std::size_t>(c));
        }
        if (op_ == "volAverage") return acc/vol;
        if (op_ == "min") return mn;
        if (op_ == "max") return mx;
        return acc;
    }
};

class SurfaceFieldValue : public Object
{
public:
    SurfaceFieldValue(const std::string& nm, const FoamDict& d, Runtime& rt)
    {
        name = nm;
        const std::string type = d.wordOr("regionType", "");
        if (type != "patch")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has regionType `" + type
                                     + "`; brae has patch for surfaceFieldValue");
        }
        patchName_ = d.wordOr("name", "");
        const std::vector<FvPatch>& patches = *rt.svc.patches;
        patch_ = -1;
        for (std::size_t i = 0; i < patches.size(); ++i)
        {
            if (patches[i].name == patchName_)
            {
                patch_ = static_cast<int>(i);
            }
        }
        if (patch_ < 0)
        {
            throw std::runtime_error("brae function objects: `" + name + "` names the patch `" + patchName_
                                     + "`, which the mesh does not have");
        }
        for (scalar a : patches[static_cast<std::size_t>(patch_)].magSf)
        {
            area_ += a;
        }
        nFaces_ = static_cast<label>(patches[static_cast<std::size_t>(patch_)].magSf.size());
        op_ = d.wordOr("operation", "");
        if (op_ != "areaAverage" && op_ != "areaIntegrate" && op_ != "sum" && op_ != "min" && op_ != "max")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has operation `" + op_
                                     + "`; brae has areaAverage, areaIntegrate, sum, min and max");
        }
        fields_ = readFieldNames(d, "fields");
        if (fields_.empty())
        {
            throw std::runtime_error("brae function objects: `" + name + "` lists no fields");
        }
        for (const std::string& f : fields_)
        {
            if (f != rt.svc.phaseField)
            {
                throw std::runtime_error("brae function objects: `" + name + "` takes the field `" + f
                                         + "`; only " + rt.svc.phaseField + " has patch values here");
            }
        }
        writeToFile_ = d.switchOr("writeToFile", true);
        if (d.switchOr("writeFields", false))
        {
            throw std::runtime_error("brae function objects: `" + name + "` sets writeFields; brae does not write "
                                     "the region's fields");
        }
        refuseKey(d, name, "weightField", "weighted operations are not ported");
        refuseKey(d, name, "postOperation", "post operations are not ported");
    }

    void open(Runtime& rt)
    {
        if (!writeToFile_)
        {
            return;
        }
        dat_ = DatFile(rt.dir(name), "surfaceFieldValue.dat");
        dat_.header("# Region type : patch " + patchName_);
        dat_.header("# Faces       : " + std::to_string(nFaces_));
        dat_.header("# Area        : " + fmtNum(area_, rt.svc.precision));
        std::string h = "# Time       ";
        for (const std::string& f : fields_)
        {
            h += " " + op_ + "(" + f + ")";
        }
        dat_.header(h);
    }

    void write(const Step& s, Runtime& rt) override
    {
        const std::vector<scalar>& a = (*rt.svc.patches)[static_cast<std::size_t>(patch_)].magSf;
        std::string line = fmtNum(s.time, rt.svc.precision);
        for (const std::string& f : fields_)
        {
            const std::vector<scalar>& v = rt.svc.src.alphaPatch(static_cast<std::size_t>(patch_));
            if (v.size() != a.size())
            {
                throw std::runtime_error("brae function objects: `" + name + "`: the patch values of " + f
                                         + " have " + std::to_string(v.size()) + " faces, the patch has "
                                         + std::to_string(a.size()));
            }
            scalar acc = 0;
            scalar mn = std::numeric_limits<scalar>::max();
            scalar mx = -std::numeric_limits<scalar>::max();
            for (std::size_t i = 0; i < a.size(); ++i)
            {
                acc += (op_ == "areaAverage" || op_ == "areaIntegrate") ? a[i]*v[i] : v[i];
                mn = std::min(mn, v[i]);
                mx = std::max(mx, v[i]);
            }
            scalar value = acc;
            if (op_ == "areaAverage") value = acc/area_;
            if (op_ == "min") value = mn;
            if (op_ == "max") value = mx;
            rt.results[name][op_ + "(" + f + ")"] = value;
            line += "\t" + fmtNum(value, rt.svc.precision);
        }
        if (writeToFile_)
        {
            dat_.row(line);
        }
    }

private:
    std::string               patchName_, op_;
    int                       patch_ = -1;
    std::vector<std::string>  fields_;
    scalar                    area_ = 0;
    label                     nFaces_ = 0;
    bool                      writeToFile_ = true;
    DatFile                   dat_;
};

// OpenFOAM's polyMesh::findCell with its default decomposition (CELL_TETS), for a handful of points. A cell is
// split into tetrahedra -- the cell centre, the face centre and each edge of the face -- and holds the points that
// lie in one of them; a point on a face shared by two cells belongs to both, and the tolerance admits it. The face
// planes (FACE_PLANES) are not used: a snappyHexMesh cell with a concave face fails that test for points inside it.
// Tried on the nearest cell centre first, then on the cells whose centres are close, then on all of them.
inline std::vector<label> locatePoints(
    const std::vector<vector>& pts,
    const PrimitiveMesh&       m,
    const FvGeometry&          g,
    const std::string&         object)
{
    const label nC = m.nCells();
    const label nF = m.nFaces();
    const label nI = m.nInternalFaces();
    const std::vector<label>& owner = m.owner();
    const std::vector<label>& nei   = m.neighbour();
    const std::vector<vector>& P    = m.points();
    std::vector<label> start(static_cast<std::size_t>(nC) + 1, 0);
    for (label f = 0; f < nF; ++f) ++start[static_cast<std::size_t>(owner[f]) + 1];
    for (label f = 0; f < nI; ++f) ++start[static_cast<std::size_t>(nei[f]) + 1];
    for (label c = 0; c < nC; ++c) start[static_cast<std::size_t>(c) + 1] += start[static_cast<std::size_t>(c)];
    std::vector<label> faces(static_cast<std::size_t>(start[static_cast<std::size_t>(nC)]));
    std::vector<label> cursor(start.begin(), start.end() - 1);
    for (label f = 0; f < nF; ++f) faces[static_cast<std::size_t>(cursor[static_cast<std::size_t>(owner[f])]++)] = f;
    for (label f = 0; f < nI; ++f) faces[static_cast<std::size_t>(cursor[static_cast<std::size_t>(nei[f])]++)] = f;

    auto vol = [](const vector& a, const vector& b, const vector& c, const vector& d)
    {
        return dot(b - a, cross(c - a, d - a));
    };
    // p inside the tetrahedron (a b c d) of either orientation, a point on a face of it included
    auto inTet = [&](const vector& a, const vector& b, const vector& c, const vector& d, const vector& p)
    {
        const scalar v0 = vol(a, b, c, d);
        if (v0 == scalar(0))
        {
            return false;
        }
        const scalar s = v0 > scalar(0) ? scalar(1) : scalar(-1);
        const scalar eps = scalar(1e-9)*std::fabs(v0);
        return s*vol(p, b, c, d) >= -eps && s*vol(a, p, c, d) >= -eps
            && s*vol(a, b, p, d) >= -eps && s*vol(a, b, c, p) >= -eps;
    };
    auto inCell = [&](const vector& p, label c)
    {
        const vector& cc = g.C()[static_cast<std::size_t>(c)];
        for (label k = start[static_cast<std::size_t>(c)]; k < start[static_cast<std::size_t>(c) + 1]; ++k)
        {
            const label f = faces[static_cast<std::size_t>(k)];
            const vector& fc = g.Cf()[static_cast<std::size_t>(f)];
            const label n = m.faceSize(f);
            for (label i = 0; i < n; ++i)
            {
                if (inTet(cc, P[static_cast<std::size_t>(m.faceVert(f, i))],
                          P[static_cast<std::size_t>(m.faceVert(f, (i + 1) % n))], fc, p))
                {
                    return true;
                }
            }
        }
        return false;
    };

    std::vector<label> out;
    for (const vector& p : pts)
    {
        label best = -1;
        scalar bd = std::numeric_limits<scalar>::max();
        for (label c = 0; c < nC; ++c)
        {
            const vector d = g.C()[static_cast<std::size_t>(c)] - p;
            const scalar dd = dot(d, d);
            if (dd < bd)
            {
                bd = dd;
                best = c;
            }
        }
        label found = (best >= 0 && inCell(p, best)) ? best : -1;
        if (found < 0 && best >= 0)
        {
            // the cells whose centres are within three cell sizes of the point, in index order
            const scalar R = scalar(3)*std::cbrt(g.V()[static_cast<std::size_t>(best)]);
            for (label c = 0; c < nC && found < 0; ++c)
            {
                const vector d = g.C()[static_cast<std::size_t>(c)] - p;
                if (dot(d, d) < R*R && inCell(p, c))
                {
                    found = c;
                }
            }
        }
        for (label c = 0; c < nC && found < 0; ++c)
        {
            if (inCell(p, c))
            {
                found = c;
            }
        }
        // found < 0: in no cell -- the caller says what to do about it
        out.push_back(found);
    }
    return out;
}

class Probes : public Object
{
public:
    Probes(const std::string& nm, const FoamDict& d, Runtime& rt)
    {
        name = nm;
        if (!d.switchOr("fixedLocations", true))
        {
            throw std::runtime_error("brae function objects: `" + name + "` has fixedLocations false; the probes "
                                     "are fixed points here");
        }
        const std::string scheme = d.wordOr("interpolationScheme", "cell");
        if (scheme != "cell")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has interpolationScheme " + scheme
                                     + "; brae reads the cell value (cell)");
        }
        fields_ = readFieldNames(d, "fields");
        for (const std::string& f : fields_)
        {
            if (f != rt.svc.phaseField && f != "p_rgh" && f != "U")
            {
                throw std::runtime_error("brae function objects: `" + name + "` probes `" + f
                                         + "`; brae probes " + rt.svc.phaseField + ", p_rgh and U");
            }
        }
        refuseKey(d, name, "sourceName", "probing a surface is not ported");
        refuseKey(d, name, "patchName", "probing a patch is not ported");
        const std::vector<scalar> xyz = d.scalarListOr("probeLocations", std::vector<scalar>());
        if (xyz.empty() || xyz.size() % 3 != 0)
        {
            throw std::runtime_error("brae function objects: `" + name + "` has no probeLocations, or a list that "
                                     "is not made of points");
        }
        for (std::size_t i = 0; i + 2 < xyz.size(); i += 3)
        {
            points_.push_back(vector{xyz[i], xyz[i + 1], xyz[i + 2]});
        }
        // OpenFOAM's probes skip a location that lies in no cell ("Did not find location ... in any cell. Skipping
        // location.") and write the files for the others, numbered from 0 again: the columns after a skipped probe
        // shift down. brae does the same, and says which locations it left out.
        {
            const std::vector<label> found = locatePoints(points_, *rt.svc.mesh, *rt.svc.geom, name);
            std::vector<vector> kept;
            std::vector<label>  keptCells;
            for (std::size_t i = 0; i < points_.size(); ++i)
            {
                if (found[i] < 0)
                {
                    noticeIgnored("functions/" + name + " probe " + std::to_string(i) + " at ("
                                  + fmtNum(points_[i].x, 8) + " " + fmtNum(points_[i].y, 8) + " "
                                  + fmtNum(points_[i].z, 8) + ")",
                                  "lies in no cell; skipped, as OpenFOAM's probes skip it. The files have one column "
                                  "fewer, and the probes after this one are numbered one lower");
                }
                else
                {
                    kept.push_back(points_[i]);
                    keptCells.push_back(found[i]);
                }
            }
            if (kept.empty())
            {
                throw std::runtime_error("brae function objects: none of the locations of `" + name
                                         + "` lies in a cell of the mesh");
            }
            points_ = kept;
            cells_ = keptCells;
        }
        for (const std::string& f : fields_)
        {
            DatFile df(rt.dir(name), f);
            for (std::size_t i = 0; i < points_.size(); ++i)
            {
                df.header("# Probe " + std::to_string(i) + " (" + fmtNum(points_[i].x, rt.svc.precision) + " "
                          + fmtNum(points_[i].y, rt.svc.precision) + " " + fmtNum(points_[i].z, rt.svc.precision) + ")");
            }
            std::string h = "#               Probe";
            for (std::size_t i = 0; i < points_.size(); ++i)
            {
                char b[32];
                std::snprintf(b, sizeof b, " %13zu", i);
                h += b;
            }
            df.header(h);
            df.header("#                Time");
            files_.push_back(std::move(df));
        }
    }

    void write(const Step& s, Runtime& rt) override
    {
        for (std::size_t k = 0; k < fields_.size(); ++k)
        {
            const std::string& f = fields_[k];
            std::string line = fmtNum(s.time, rt.svc.precision);
            if (f == "U")
            {
                const std::vector<scalar>& ux = rt.svc.src.Ux();
                const std::vector<scalar>& uy = rt.svc.src.Uy();
                const std::vector<scalar>& uz = rt.svc.src.Uz();
                for (label c : cells_)
                {
                    const std::size_t i = static_cast<std::size_t>(c);
                    line += " (" + fmtNum(ux[i], rt.svc.precision) + " " + fmtNum(uy[i], rt.svc.precision) + " "
                          + fmtNum(uz[i], rt.svc.precision) + ")";
                }
            }
            else
            {
                const std::vector<scalar>& v = (f == "p_rgh") ? rt.svc.src.p_rgh() : rt.svc.src.alpha();
                for (label c : cells_)
                {
                    line += " " + fmtNum(v[static_cast<std::size_t>(c)], rt.svc.precision);
                }
            }
            files_[k].row(line);
        }
    }

private:
    std::vector<std::string> fields_;
    std::vector<vector>      points_;
    std::vector<label>       cells_;
    std::vector<DatFile>     files_;
};

class FieldMinMax : public Object
{
public:
    FieldMinMax(const std::string& nm, const FoamDict& d, Runtime& rt)
    {
        name = nm;
        fields_ = readFieldNames(d, "fields");
        mode_ = d.wordOr("mode", "magnitude");
        if (mode_ != "magnitude" && mode_ != "component")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has mode " + mode_
                                     + "; brae has magnitude and component");
        }
        for (const std::string& f : fields_)
        {
            if (f != "U" && !rt.isScalarName(f))
            {
                throw std::runtime_error("brae function objects: `" + name + "` takes the field `" + f
                                         + "`, which is not one brae knows");
            }
        }
        refuseKey(d, name, "location", "only the default (location written) is ported");
        dat_ = DatFile(rt.dir(name), "fieldMinMax.dat");
        dat_.header("# Field minima and maxima");
        dat_.header("# Time          field           min             location(min)           processor       "
                    "max             location(max)           processor");
    }

    void write(const Step& s, Runtime& rt) override
    {
        const std::vector<vector>& C = rt.svc.geom->C();
        auto vec3 = [&](std::size_t c)
        {
            return "(" + fmtNum(C[c].x, rt.svc.precision) + " " + fmtNum(C[c].y, rt.svc.precision) + " "
                 + fmtNum(C[c].z, rt.svc.precision) + ")";
        };
        for (const std::string& f : fields_)
        {
            auto emit = [&](const std::string& label, const std::vector<scalar>& val)
            {
                std::size_t imin = 0, imax = 0;
                for (std::size_t i = 1; i < val.size(); ++i)
                {
                    if (val[i] < val[imin]) imin = i;
                    if (val[i] > val[imax]) imax = i;
                }
                dat_.row(fmtNum(s.time, rt.svc.precision) + "\t" + label + "\t" + fmtNum(val[imin], rt.svc.precision)
                         + "\t" + vec3(imin) + "\t0\t" + fmtNum(val[imax], rt.svc.precision) + "\t" + vec3(imax) + "\t0");
            };
            if (f == "U")
            {
                const std::vector<scalar>& ux = rt.svc.src.Ux();
                const std::vector<scalar>& uy = rt.svc.src.Uy();
                const std::vector<scalar>& uz = rt.svc.src.Uz();
                if (mode_ == "magnitude")
                {
                    std::vector<scalar> m(ux.size());
                    for (std::size_t i = 0; i < m.size(); ++i)
                    {
                        m[i] = std::sqrt(ux[i]*ux[i] + uy[i]*uy[i] + uz[i]*uz[i]);
                    }
                    emit("mag(U)", m);
                }
                else
                {
                    emit("U.x", ux);
                    emit("U.y", uy);
                    emit("U.z", uz);
                }
            }
            else
            {
                const SF sf = rt.scalarField(f);
                if (sf.isConst())
                {
                    emit(f, std::vector<scalar>(rt.nCells(), sf.c));
                }
                else
                {
                    emit(f, sf.v());
                }
            }
        }
    }

private:
    std::vector<std::string> fields_;
    std::string              mode_;
    DatFile                  dat_;
};

class ExprField : public Object
{
public:
    ExprField(const std::string& nm, const FoamDict& d, Runtime& rt)
    {
        name = nm;
        const std::string field = d.wordOr("field", "");
        if (field.empty())
        {
            throw std::runtime_error("brae function objects: `" + name + "` names no field");
        }
        const std::string action = d.wordOr("action", "new");
        if (action != "new")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has action " + action
                                     + "; brae has new");
        }
        const auto* e = d.find("expression");
        if (!e || e->empty())
        {
            throw std::runtime_error("brae function objects: `" + name + "` has no expression");
        }
        std::string text;
        for (const std::string& t : *e)
        {
            text += t;
        }
        // strip a quote pair the dictionary reader kept
        if (text.size() >= 2 && text.front() == '"' && text.back() == '"')
        {
            text = text.substr(1, text.size() - 2);
        }
        if (d.found("autowrite") && d.switchOr("autowrite", false))
        {
            throw std::runtime_error("brae function objects: `" + name + "` has autowrite true; brae keeps expression "
                                     "fields in memory and does not write them");
        }
        if (!d.found("autowrite"))
        {
            noticeDefaulted("functions/" + name + " autowrite", "false: brae does not write expression fields");
        }
        d.find("dimensions");   // read, and unused: the evaluator is not dimensioned
        rt.exprs[field] = text;
        // evaluate once at start-up on zero-size views? No: a syntax error should stop the run now, so parse it
        {
            Scope sc;
            sc.n = 1;
            static const std::vector<scalar> one(1, 1.0);
            sc.scalarField = [&](const std::string& nm, SF& o)
            {
                if (rt.isScalarName(nm) || nm == field)
                {
                    o = sfBorrow(one);
                    return true;
                }
                return false;
            };
            sc.vectorField = [&](const std::string& nm, VF& o)
            {
                if (nm != "U")
                {
                    return false;
                }
                o.x = o.y = o.z = sfBorrow(one);
                return true;
            };
            sc.pos = [&]() { VF v; v.x = v.y = v.z = sfBorrow(one); return v; };
            ExprParser(text, sc).parseScalar();
        }
    }

    void write(const Step&, Runtime&) override {}   // evaluated where it is read
};

class RunTimeControl : public Object
{
public:
    RunTimeControl(const std::string& nm, const FoamDict& d, Runtime&)
    {
        name = nm;
        action_ = d.wordOr("satisfiedAction", "end");
        if (action_ != "end")
        {
            throw std::runtime_error("brae function objects: `" + name + "` has satisfiedAction " + action_
                                     + "; brae has end");
        }
        const FoamDict* conds = d.subDict("conditions");
        if (!conds || conds->subs.empty())
        {
            throw std::runtime_error("brae function objects: `" + name + "` has no conditions");
        }
        for (const auto& c : conds->subs)
        {
            Cond cd;
            cd.name = c.first;
            const std::string type = c.second.wordOr("type", "");
            if (type != "minMax")
            {
                throw std::runtime_error("brae function objects: `" + name + "` condition `" + c.first
                                         + "` has type " + type + "; brae has minMax");
            }
            cd.object = c.second.wordOr("functionObject", "");
            cd.fields = readFieldNames(c.second, "fields");
            const std::string mode = c.second.wordOr("mode", "");
            if (mode != "maximum" && mode != "minimum")
            {
                throw std::runtime_error("brae function objects: `" + name + "` condition `" + c.first
                                         + "` has mode " + mode + "; brae has maximum and minimum");
            }
            cd.maximum = (mode == "maximum");
            cd.threshold = c.second.scalarOr("value", std::numeric_limits<scalar>::quiet_NaN());
            if (cd.object.empty() || cd.fields.empty() || std::isnan(cd.threshold))
            {
                throw std::runtime_error("brae function objects: `" + name + "` condition `" + c.first
                                         + "` needs functionObject, fields and value");
            }
            conds_.push_back(cd);
        }
    }

    bool canEnd() const { return true; }
    bool evaluatesEveryStep() const override { return true; }
    void write(const Step&, Runtime&) override {}

    bool evaluate(const Step&, Runtime& rt) override
    {
        const int prec = std::max(rt.svc.precision, 6);
        if (log)
        {
            std::printf("runTimeControl %s output:\n", name.c_str());
        }
        bool met = true;
        for (const Cond& c : conds_)
        {
            scalar value = c.maximum ? -std::numeric_limits<scalar>::max() : std::numeric_limits<scalar>::max();
            for (const std::string& f : c.fields)
            {
                const scalar v = rt.result(c.object, f);
                value = c.maximum ? std::max(value, v) : std::min(value, v);
            }
            const bool satisfied = c.maximum ? (value > c.threshold) : (value < c.threshold);
            met = met && satisfied;
            if (log)
            {
                std::printf("    minMax: %s %s: value = %s, threshold value = %s, satisfied = %s\n",
                            c.maximum ? "maximum" : "minimum", c.fields.front().c_str(), fmtNum(value, prec).c_str(),
                            fmtNum(c.threshold, prec).c_str(), satisfied ? "true" : "false");
            }
        }
        if (log)
        {
            std::printf(met ? "    Conditions met\n" : "    Conditions not met\n");
            std::fflush(stdout);
        }
        return met;   // satisfiedAction end
    }

private:
    struct Cond
    {
        std::string               name, object;
        std::vector<std::string>  fields;
        bool                      maximum = true;
        scalar                    threshold = 0;
    };
    std::string       action_;
    std::vector<Cond> conds_;
};

// ---------------------------------------------------------------------------------------------------------------
// the list
// ---------------------------------------------------------------------------------------------------------------

class FunctionObjectList
{
public:
    // Builds the list from controlDict `functions`; null when it has none that brae runs.
    static std::shared_ptr<FunctionObjectList> read(const FoamDict& controlDict, Services svc)
    {
        const FoamDict* fns = controlDict.subDict("functions");
        if (!fns || fns->subs.empty())
        {
            return nullptr;
        }
        std::shared_ptr<FunctionObjectList> list(new FunctionObjectList(std::move(svc)));
        Runtime& rt = list->rt_;
        std::vector<std::string> notRun;
        for (const auto& e : fns->subs)
        {
            const FoamDict& d = e.second;
            const std::string type = d.wordOr("type", "");
            const bool enabled = d.switchOr("enabled", true);
            d.find("libs");
            std::unique_ptr<Object> o;
            if (type == "volFieldValue" || type == "surfaceFieldValue" || type == "probes" || type == "fieldMinMax"
             || type == "exprField" || type == "runTimeControl")
            {
                if (!enabled)
                {
                    list->disabled_.push_back(e.first);
                    continue;
                }
                refuseKey(d, e.first, "timeStart", "an active window (timeStart/timeEnd) is not ported");
                refuseKey(d, e.first, "timeEnd", "an active window (timeStart/timeEnd) is not ported");
                if (type == "volFieldValue")
                {
                    auto v = std::unique_ptr<VolFieldValue>(new VolFieldValue(e.first, d, rt));
                    v->open(rt);
                    o = std::move(v);
                }
                else if (type == "surfaceFieldValue")
                {
                    auto v = std::unique_ptr<SurfaceFieldValue>(new SurfaceFieldValue(e.first, d, rt));
                    v->open(rt);
                    o = std::move(v);
                }
                else if (type == "probes")
                {
                    o.reset(new Probes(e.first, d, rt));
                }
                else if (type == "fieldMinMax")
                {
                    o.reset(new FieldMinMax(e.first, d, rt));
                }
                else if (type == "exprField")
                {
                    o.reset(new ExprField(e.first, d, rt));
                }
                else
                {
                    o.reset(new RunTimeControl(e.first, d, rt));
                    list->canStop_ = true;
                }
                o->log = d.switchOr("log", true);
                o->ctl = Control::read(d, e.first, "write");
                list->objects_.push_back(std::move(o));
                list->ran_.push_back(e.first);
            }
            else
            {
                notRun.push_back(e.first + " (" + type + ")");
            }
        }
        std::string line = "  function objects: run";
        for (const std::string& n : list->ran_) line += " `" + n + "`";
        if (list->ran_.empty()) line += " none";
        if (!list->disabled_.empty())
        {
            line += "; disabled";
            for (const std::string& n : list->disabled_) line += " `" + n + "`";
        }
        std::printf("%s\n", line.c_str());
        if (!notRun.empty())
        {
            std::string nr;
            for (const std::string& n : notRun) nr += " " + n;
            noticeIgnored("controlDict functions", "not run by brae (a type it does not port):" + nr);
        }
        if (list->objects_.empty())
        {
            return nullptr;
        }
        return list;
    }

    bool canStop() const { return canStop_; }

    // One call per completed step. True when a runTimeControl asks the run to end at this step.
    bool execute(const Step& s)
    {
        ++rt_.stamp;
        if (rt_.svc.src.newStep)
        {
            rt_.svc.src.newStep();
        }
        for (auto& o : objects_)
        {
            if (!o->evaluatesEveryStep() && o->ctl.due(s))
            {
                o->write(s, rt_);
            }
        }
        bool stop = false;
        for (auto& o : objects_)
        {
            if (o->evaluatesEveryStep() && o->evaluate(s, rt_))
            {
                stop = true;
            }
        }
        return stop;
    }

private:
    explicit FunctionObjectList(Services svc) : rt_(std::move(svc)) {}

    Runtime                               rt_;
    std::vector<std::unique_ptr<Object>>  objects_;
    std::vector<std::string>              ran_, disabled_;
    bool                                  canStop_ = false;
};

// What both time loops call once, before the first step: the list for the case's own controlDict, or null when
// it has no function object brae runs. `src` is the loop's way of showing its fields on the host.
inline std::shared_ptr<FunctionObjectList> createFunctionObjects(
    const std::string&                                caseDir,
    const std::string&                                startDir,
    const PrimitiveMesh&                              m,
    const FvGeometry&                                 g,
    const std::vector<FvPatch>&                       patches,
    const std::map<std::string, std::vector<label>>&  zones,
    const std::string&                                phaseField,
    bool                                              meshChanges,
    const Source&                                     src)
{
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    Services s;
    s.caseDir = caseDir;
    s.startName = std::filesystem::path(startDir).filename().string();
    s.precision = std::max(controlDict.intOr("writePrecision", 6), 6);
    s.phaseField = phaseField;
    s.mesh = &m;
    s.geom = &g;
    s.patches = &patches;
    s.cellZones = &zones;
    s.src = src;
    std::shared_ptr<FunctionObjectList> list = FunctionObjectList::read(controlDict, s);
    if (list && meshChanges)
    {
        throw std::runtime_error(
            "brae interFoam: the case has function objects brae runs and a mesh that moves or refines. They are "
            "evaluated on the start mesh's cells and faces; on a changing mesh that would be a wrong answer, so "
            "the run stops here. Remove the function objects, or run the case with a static mesh.");
    }
    return list;
}

}   // namespace fo
}   // namespace brae
