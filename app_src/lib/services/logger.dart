// For debugging purposes

class Logger {
  bool showLogs = false;

  Logger({this.showLogs});

  void call(String message) {
    if (showLogs) {
      print(message);
    }
  }
}
