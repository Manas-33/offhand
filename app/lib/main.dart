import 'package:flutter/material.dart';

import 'screens/home_screen.dart';

void main() => runApp(const OffhandApp());

class OffhandApp extends StatelessWidget {
  const OffhandApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Offhand',
      theme: ThemeData(colorSchemeSeed: Colors.teal, useMaterial3: true),
      home: const HomeScreen(),
    );
  }
}
