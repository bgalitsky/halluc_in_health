import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.Objects;

public final class DiscourseParser {

    private static final String[] BUILDER_CLASS_CANDIDATES = new String[] {
            "opennlp.tools.parse_thicket.ParseCorefBuilderWithNERandRST",
            "opennlp.tools.parse_thicket.external_rst.ParseCorefBuilderWithNERandRST"
    };

    private final Object builder;
    private final Method buildMethod;

    public DiscourseParser(boolean disableSUTime) {
        if (disableSUTime) {
            System.setProperty("sutime.binders", "0");
            System.setProperty("ner.applyNumericClassifiers", "false");
        }

        Class<?> builderClass = loadFirstAvailable(BUILDER_CLASS_CANDIDATES);
        if (builderClass == null) {
            throw new IllegalStateException(
                    "Could not find ParseCorefBuilderWithNERandRST in classpath. Add cb_0.11.jar."
            );
        }

        // 1) public static getInstance()
        Object b = tryStaticFactory(builderClass, "getInstance");
        if (b == null) {
            // 2) non-public static getInstance()
            b = tryStaticFactoryNonPublic(builderClass, "getInstance");
        }
        if (b == null) {
            // 3) public no-arg ctor
            b = tryCtor(builderClass);
        }
        if (b == null) {
            // 4) non-public no-arg ctor (make accessible)
            b = tryCtorNonPublic(builderClass);
        }
        if (b == null) {
            throw new RuntimeException("Unable to instantiate builder " + builderClass.getName());
        }
        this.builder = Objects.requireNonNull(b, "Builder instance is null");

        try {
            this.buildMethod = builderClass.getMethod("buildParseThicket", String.class);
        } catch (Exception e) {
            throw new RuntimeException("Builder lacks buildParseThicket(String).", e);
        }
    }

    /** Parse the text and return the discourse tree as a String. */
    public String parseToRST(String text) {
        if (text == null || text.trim().isEmpty()) return "";
        Object result;
        try {
            result = buildMethod.invoke(builder, text);
        } catch (Exception e) {
            throw new RuntimeException("Error invoking buildParseThicket(String).", e);
        }
        if (result == null) return "";

        // Prefer getDtDump()
        String dump = tryCallStringNoArg(result, "getDtDump");
        if (dump != null && !dump.trim().isEmpty()) return dump.trim();

        // Else getDt().toString()
        Object dt = tryCallNoArg(result, "getDt");
        if (dt != null) {
            String s = String.valueOf(dt);
            if (s != null && !s.trim().isEmpty()) return s.trim();
        }

        // Fallback
        String s = String.valueOf(result);
        return s == null ? "" : s.trim();
    }

    // ---------- Reflection helpers ----------

    private static Class<?> loadFirstAvailable(String[] candidates) {
        for (String fqn : candidates) {
            try { return Class.forName(fqn); }
            catch (ClassNotFoundException ignore) {}
        }
        return null;
    }

    private static Object tryStaticFactory(Class<?> cls, String name) {
        try {
            Method m = cls.getMethod(name);
            if ((m.getModifiers() & java.lang.reflect.Modifier.STATIC) == 0) return null;
            return m.invoke(null);
        } catch (NoSuchMethodException e) {
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    private static Object tryStaticFactoryNonPublic(Class<?> cls, String name) {
        try {
            Method m = cls.getDeclaredMethod(name);
            m.setAccessible(true);
            if ((m.getModifiers() & java.lang.reflect.Modifier.STATIC) == 0) return null;
            return m.invoke(null);
        } catch (NoSuchMethodException e) {
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    private static Object tryCtor(Class<?> cls) {
        try {
            Constructor<?> c = cls.getConstructor();
            return c.newInstance();
        } catch (NoSuchMethodException e) {
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    private static Object tryCtorNonPublic(Class<?> cls) {
        try {
            Constructor<?> c = cls.getDeclaredConstructor();
            c.setAccessible(true);
            return c.newInstance();
        } catch (NoSuchMethodException e) {
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    private static Object tryCallNoArg(Object target, String method) {
        try {
            Method m = target.getClass().getMethod(method);
            m.setAccessible(true);
            return m.invoke(target);
        } catch (NoSuchMethodException e) {
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    private static String tryCallStringNoArg(Object target, String method) {
        Object o = tryCallNoArg(target, method);
        return (o == null) ? null : String.valueOf(o);
    }

    // ---------- CLI ----------

    private static void usage() {
        System.out.println(
                "DiscourseParser\n\n" +
                        "Usage:\n" +
                        "  java -cp \".;cb_0.11.jar\" DiscourseParser \"your text here\"\n" +
                        "  java -cp \".;cb_0.11.jar\" DiscourseParser --file input.txt\n" +
                        "  java -cp \".;cb_0.11.jar\" DiscourseParser --stdin\n" +
                        "  Add --no-sutime to try disabling SUTime/JollyDay.\n"
        );
    }

    public static void main(String[] args) throws Exception {
        boolean disableSUTime = false;
        int idx = 0;
        while (idx < args.length && args[idx].startsWith("--")) {
            if ("--no-sutime".equalsIgnoreCase(args[idx])) disableSUTime = true;
            else if ("--help".equalsIgnoreCase(args[idx]) || "-h".equalsIgnoreCase(args[idx])) { usage(); return; }
            else if ("--file".equalsIgnoreCase(args[idx]) || "--stdin".equalsIgnoreCase(args[idx])) break;
            else { System.err.println("Unknown option: " + args[idx]); usage(); return; }
            idx++;
        }
        DiscourseParser parser = new DiscourseParser(disableSUTime);

        String output;
        if (idx < args.length && "--file".equalsIgnoreCase(args[idx])) {
            if (idx + 1 >= args.length) { System.err.println("Missing filename after --file"); usage(); return; }
            String path = args[idx + 1];
            StringBuilder sb = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new FileReader(path))) {
                String line; while ((line = br.readLine()) != null) sb.append(line).append('\n');
            }
            output = parser.parseToRST(sb.toString());
        } else if (idx < args.length && "--stdin".equalsIgnoreCase(args[idx])) {
            StringBuilder sb = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
                String line; while ((line = br.readLine()) != null) sb.append(line).append('\n');
            }
            output = parser.parseToRST(sb.toString());
        } else if (idx < args.length) {
            StringBuilder sb = new StringBuilder();
            for (int i = idx; i < args.length; i++) { if (i > idx) sb.append(' '); sb.append(args[i]); }
            output = parser.parseToRST(sb.toString());
        } else { usage(); return; }

        System.out.println(output == null ? "" : output);
    }
}
