import React, { useState, useEffect } from "react";
import { 
  Scan, 
  ArrowRight, 
  Eye, 
  EyeOff, 
  Lock, 
  Mail, 
  User, 
  AlertCircle, 
  AlertTriangle,
  Loader2
} from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function SignupPage({ onNavigate }) {
  const { 
    signup, 
    loginWithGoogle, 
    authError, 
    setAuthError, 
    isFirebaseConfigured, 
    missingConfigKeys 
  } = useAuth();

  useEffect(() => {
    setAuthError(null);
  }, [setAuthError]);

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [agreeTerms, setAgreeTerms] = useState(true);
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);

  // Compute password strength
  const getPasswordStrength = () => {
    if (!password) return { label: "", score: 0, color: "" };
    let score = 0;
    if (password.length >= 6) score += 1;
    if (password.length >= 10) score += 1;
    if (/[0-9]/.test(password)) score += 1;
    if (/[^A-Za-z0-9]/.test(password)) score += 1;

    if (score <= 1) return { label: "Weak", score: 1, color: "bg-red-500" };
    if (score <= 3) return { label: "Moderate", score: 2, color: "bg-[#f5a524]" };
    return { label: "Strong", score: 3, color: "bg-emerald-500" };
  };

  const strength = getPasswordStrength();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) {
      setAuthError("Email and password are required.");
      return;
    }

    if (password.length < 6) {
      setAuthError("Password must be at least 6 characters in length.");
      return;
    }

    if (password !== confirmPassword) {
      setAuthError("Passwords do not match. Please verify.");
      return;
    }

    if (!agreeTerms) {
      setAuthError("Please agree to the RoadVision platform terms to proceed.");
      return;
    }

    try {
      setLoading(true);
      await signup(email, password, name);
      onNavigate("dashboard");
    } catch (err) {
      // Error handled in AuthContext
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignUp = async () => {
    try {
      setGoogleLoading(true);
      const user = await loginWithGoogle();
      if (user) {
        onNavigate("dashboard");
      }
    } catch (err) {
      // Handled via authError
    } finally {
      setGoogleLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0b] text-[#f3f1ec] flex flex-col justify-between selection:bg-[#f5a524] selection:text-[#0a0a0b]">
      {/* Top Bar */}
      <header className="border-b border-white/10 px-6 py-4 flex items-center justify-between">
        <button 
          type="button"
          className="flex items-center gap-3 cursor-pointer text-left"
          onClick={() => onNavigate("landing")}
        >
          <div className="w-8 h-8 rounded-full bg-[#f5a524] flex items-center justify-center font-bold text-[#0a0a0b]">
            <Scan className="w-4 h-4" />
          </div>
          <span className="font-bold tracking-tight text-white">RoadVision<span className="text-[#f5a524]">.AI</span></span>
        </button>

        <button 
          type="button"
          onClick={() => onNavigate("landing")}
          className="text-xs font-mono text-white/60 hover:text-white"
        >
          ← Back to Overview
        </button>
      </header>

      {/* Main Registration Card */}
      <main className="flex-1 flex items-center justify-center p-4 sm:p-6 my-6">
        <div className="w-full max-w-md bg-[#121214] border border-white/10 rounded-2xl p-6 sm:p-8 shadow-2xl relative">
          {/* Header */}
          <div className="mb-6">
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-white/5 border border-white/10 text-[#f5a524] text-[10px] font-mono tracking-widest uppercase mb-3">
              INSTITUTIONAL REGISTRATION
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white mb-2">
              Create Your Account
            </h1>
            <p className="text-sm text-white/60">
              Start monitoring road conditions with intelligent computer vision.
            </p>
          </div>

          {/* Firebase Configuration Status Notice */}
          {!isFirebaseConfigured && (
            <div className="mb-5 p-3.5 rounded-xl bg-amber-950/30 border border-[#f5a524]/30 text-[#f5a524] text-xs flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <div className="leading-relaxed">
                <span className="font-semibold block mb-0.5">Firebase Environment Not Configured</span>
                <span className="text-white/70">
                  Missing required variable(s): <code className="font-mono text-[#f5a524]">{missingConfigKeys?.join(", ") || "VITE_FIREBASE_API_KEY"}</code>. Set in <code className="font-mono text-[#f5a524]">frontend/.env</code> and restart Vite.
                </span>
              </div>
            </div>
          )}

          {/* Error Alert */}
          {authError && (
            <div className="mb-6 p-3.5 rounded-xl bg-red-950/40 border border-red-500/30 text-red-300 text-xs flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <span>{authError}</span>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-mono text-white/70 mb-1.5 uppercase tracking-wider">
                Full Name / Department
              </label>
              <div className="relative">
                <User className="w-4 h-4 text-white/40 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input 
                  type="text"
                  required
                  placeholder="Dr. Jordan Hayes (DOT Field Ops)"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full h-11 pl-10 pr-3.5 rounded-xl bg-[#0a0a0b] border border-white/10 text-white placeholder-white/30 text-sm focus:outline-none focus:border-[#f5a524] transition-colors"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-mono text-white/70 mb-1.5 uppercase tracking-wider">
                Work Email
              </label>
              <div className="relative">
                <Mail className="w-4 h-4 text-white/40 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input 
                  type="email"
                  required
                  placeholder="jordan.hayes@dot.state.gov"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full h-11 pl-10 pr-3.5 rounded-xl bg-[#0a0a0b] border border-white/10 text-white placeholder-white/30 text-sm focus:outline-none focus:border-[#f5a524] transition-colors"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-mono text-white/70 mb-1.5 uppercase tracking-wider">
                Password
              </label>
              <div className="relative">
                <Lock className="w-4 h-4 text-white/40 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input 
                  type={showPassword ? "text" : "password"}
                  required
                  placeholder="Min. 6 alphanumeric characters"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full h-11 pl-10 pr-10 rounded-xl bg-[#0a0a0b] border border-white/10 text-white placeholder-white/30 text-sm focus:outline-none focus:border-[#f5a524] transition-colors"
                />
                <button 
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-white/40 hover:text-white"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>

              {/* Password Strength Indicator */}
              {password && (
                <div className="mt-2 flex items-center justify-between text-[11px] font-mono">
                  <span className="text-white/50">Security Level:</span>
                  <div className="flex items-center gap-1.5">
                    <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden flex gap-0.5">
                      <div className={`h-full ${strength.color}`} style={{ width: `${(strength.score / 3) * 100}%` }} />
                    </div>
                    <span className="text-white/80">{strength.label}</span>
                  </div>
                </div>
              )}
            </div>

            <div>
              <label className="block text-xs font-mono text-white/70 mb-1.5 uppercase tracking-wider">
                Confirm Password
              </label>
              <div className="relative">
                <Lock className="w-4 h-4 text-white/40 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input 
                  type={showPassword ? "text" : "password"}
                  required
                  placeholder="Re-enter password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full h-11 pl-10 pr-3.5 rounded-xl bg-[#0a0a0b] border border-white/10 text-white placeholder-white/30 text-sm focus:outline-none focus:border-[#f5a524] transition-colors"
                />
              </div>
            </div>

            <div className="pt-1">
              <label className="flex items-start gap-2.5 cursor-pointer text-xs text-white/70">
                <input 
                  type="checkbox"
                  checked={agreeTerms}
                  onChange={(e) => setAgreeTerms(e.target.checked)}
                  className="w-3.5 h-3.5 rounded border-white/20 bg-[#0a0a0b] text-[#f5a524] focus:ring-0 mt-0.5"
                />
                <span className="leading-tight">
                  I agree to the RoadVision AI terms of service, acceptable infrastructure data use and inspection audit policies.
                </span>
              </label>
            </div>

            <button 
              type="submit"
              disabled={loading}
              className="btn-pivai-amber w-full justify-center h-11 mt-3 text-sm font-semibold"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                  Creating Workspace...
                </>
              ) : (
                <>
                  Create Account
                  <ArrowRight className="w-4 h-4 ml-1" />
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="relative my-6 text-center">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-white/10"></div>
            </div>
            <span className="relative px-3 bg-[#121214] text-[11px] font-mono text-white/40 uppercase">
              Or Register With
            </span>
          </div>

          {/* Google Sign Up */}
          <button 
            type="button"
            onClick={handleGoogleSignUp}
            disabled={googleLoading}
            className="w-full h-11 rounded-full border border-white/15 bg-white/5 hover:bg-white/10 text-white text-xs font-semibold flex items-center justify-center gap-3 transition-colors"
          >
            {googleLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <svg className="w-4 h-4" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z" />
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z" />
              </svg>
            )}
            <span>{googleLoading ? "Connecting to Google..." : "Continue with Google"}</span>
          </button>

          {/* Footer Link */}
          <div className="mt-6 text-center text-xs text-white/50">
            Already have an active account?{" "}
            <button 
              type="button"
              onClick={() => onNavigate("login")}
              className="text-[#f5a524] hover:underline font-semibold"
            >
              Sign In
            </button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 px-6 py-4 text-center text-xs text-white/40 font-mono">
        RoadVision AI · Enterprise Cloud Authentication
      </footer>
    </div>
  );
}
