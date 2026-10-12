import json
import unittest
from pathlib import Path

from text_processing import GrammarCorrector, SignTextPipeline
from text_processing.lexicon import SIGN_GLOSS_TURKISH, restore_turkish, restore_turkish_words

ROOT = Path(__file__).resolve().parents[1]


class GrammarEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = GrammarCorrector()

    def test_rule_based_sentence_assembly(self):
        self.assertEqual(self.engine.correct(["ben", "okul", "gitmek"]), "Ben okula gidiyorum.")

    def test_nominal_predicate_gets_copula(self):
        self.assertEqual(self.engine.correct(["sen", "hasta"]), "Sen hastasın.")
        self.assertEqual(self.engine.correct(["merhaba", "ben", "öğretmen"]), "Merhaba, ben öğretmenim.")

    def test_pipeline_can_disable_tts_per_request(self):
        result = SignTextPipeline().correct(["merhaba", "ben", "su", "istemek"], synthesize_audio=False)
        self.assertEqual(result.sentence, "Merhaba, ben su istiyorum.")
        self.assertEqual(result.tts_status, "disabled")
        self.assertIsNone(result.audio_path)

    def test_pipeline_waits_for_two_words(self):
        result = SignTextPipeline().correct(["ben"], synthesize_audio=False)
        self.assertEqual(result.sentence, "")


class SignLexiconTests(unittest.TestCase):
    def test_covers_every_avatar_and_model_gloss(self):
        landmark_glosses = {p.stem for p in (ROOT / "landmarks").glob("*.json")}
        labels = json.loads((ROOT / "demo_assets_179" / "label_map.json").read_text(encoding="utf-8"))
        model_glosses = {entry["TR"] for entry in labels.values()}
        self.assertEqual(len(landmark_glosses), 226)
        self.assertEqual(set(SIGN_GLOSS_TURKISH), landmark_glosses)
        self.assertTrue(model_glosses <= landmark_glosses)

    def test_spellings_are_readable_turkish(self):
        for gloss, text in SIGN_GLOSS_TURKISH.items():
            self.assertNotIn("_", text, gloss)
            self.assertTrue(text.strip(), gloss)

    def test_known_and_unknown_words(self):
        self.assertEqual(restore_turkish("cay"), "çay")
        self.assertEqual(restore_turkish("afiyet_olsun"), "afiyet olsun")
        self.assertEqual(restore_turkish("gol"), "göl")
        self.assertEqual(restore_turkish("merhaba"), "merhaba")

    def test_live_glosses_form_correct_sentences(self):
        engine = GrammarCorrector()
        cases = {
            ("biz", "gulmek"): "Biz gülüyoruz.",
            ("ben", "ozur_dilemek"): "Ben özür diliyorum.",
            ("ben", "kopek", "sevmek"): "Ben köpeği seviyorum.",
            ("onlar", "aglamak"): "Onlar ağlıyorlar.",
            ("ben", "ogretmen"): "Ben öğretmenim.",
        }
        for glosses, expected in cases.items():
            self.assertEqual(engine.correct(restore_turkish_words(glosses)), expected, glosses)


if __name__ == "__main__":
    unittest.main()
